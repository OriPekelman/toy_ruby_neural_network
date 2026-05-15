#include "tinynn_ggml.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TINYNN_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define TNN_SCRATCH_BYTES (16 * 1024 * 1024)   /* 16 MiB: 4M f32 */

/* Engine: persistent across the program's lifetime. Holds the backend
 * objects + scheduler. Cached per (prefer_cuda) flavor so multiple
 * session_new calls share one backend init. */
typedef struct {
    ggml_backend_t       backend;        /* CUDA or CPU */
    ggml_backend_t       cpu_backend;    /* sched fallback when primary is CUDA */
    ggml_backend_sched_t sched;
    const char          *backend_name;
} tnn_engine;

static tnn_engine *g_engine_cpu  = NULL;
static tnn_engine *g_engine_cuda = NULL;

static tnn_engine *tnn_engine_get(int prefer_cuda)
{
    tnn_engine **slot = prefer_cuda ? &g_engine_cuda : &g_engine_cpu;
    if (*slot) return *slot;

    ggml_backend_load_all();
    tnn_engine *e = (tnn_engine *)calloc(1, sizeof(tnn_engine));
    if (!e) return NULL;

#ifdef TINYNN_HAVE_CUDA
    if (prefer_cuda) {
        e->backend = ggml_backend_cuda_init(0);
        e->backend_name = "cuda";
    }
#endif
    if (!e->backend) {
        e->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        e->backend_name = "cpu";
    }
    if (!e->backend) { free(e); return NULL; }

    e->cpu_backend = (e->backend_name[0] == 'c' && e->backend_name[1] == 'p')
        ? NULL
        : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = e->backend;
    if (e->cpu_backend) backends[n_backends++] = e->cpu_backend;
    /* Scheduler graph-size hint. Must be >= n_nodes + n_leafs of the
     * largest graph we'll alloc. 16384 matches the per-session graph
     * cap (see tnn_session_new). */
    e->sched = ggml_backend_sched_new(backends, NULL, n_backends,
                                       16384, false, true);

    *slot = e;
    return e;
}

/* Session: per "compute frame" — owns its ctx + graph + scratch, but
 * references a cached engine. tnn_session_free frees the per-frame
 * resources only; the engine persists for reuse.
 *
 * Two contexts:
 *  - ctx_w (weights_ctx): persistent tensors (parameters, moments).
 *    Allocated once via ggml_backend_alloc_ctx_tensors into a stable
 *    backend buffer that survives sched_reset cycles.
 *  - ctx (compute_ctx): per-step tensors (inputs, intermediates).
 *    Managed by backend_sched, re-allocated per compute cycle.
 *
 * Cross-ctx tensors in a single graph are supported by ggml — nodes
 * just hold tensor pointers. The compute graph references both ctxs;
 * sched skips persistent tensors (they already have a buffer). */
typedef struct {
    tnn_engine             *engine;       /* unowned */
    struct ggml_context    *ctx;          /* compute (no_alloc=true) */
    struct ggml_context    *ctx_w;        /* weights  (no_alloc=true until finalized) */
    struct ggml_cgraph     *graph;        /* primary (e.g. forward) */
    struct ggml_cgraph     *graph_b;      /* secondary (e.g. adam_step) */
    uint8_t                *ctx_buf;
    size_t                  ctx_buf_size;
    uint8_t                *ctx_w_buf;
    size_t                  ctx_w_buf_size;
    ggml_backend_buffer_t   weights_buf;  /* set by tnn_finalize_weights */
    float                  *scratch;
    int                     realized;
    int                     realized_b;
    int                     weights_finalized;
    int                     last_graph;   /* 0 = none, 1 = a, 2 = b */
} tnn_session;

void *tnn_session_new(int prefer_cuda)
{
    tnn_engine *e = tnn_engine_get(prefer_cuda);
    if (!e) return NULL;

    tnn_session *s = (tnn_session *)calloc(1, sizeof(tnn_session));
    if (!s) return NULL;
    s->engine = e;

    /* Reset the (shared) scheduler so any prior allocation state is
     * wiped before this session builds its graph. */
    ggml_backend_sched_reset(e->sched);

    /* Two cgraphs share ctx, so reserve room for both. ctx grows
     * monotonically across tnn_reset_for_rebuild cycles (each rebuild
     * allocates new compute-tensor metadata in the same ctx). At
     * GPT-2-distil shape one decode-step graph has ~1280 ops:
     *   6 layers × (12 heads × ~16 ops + concat/proj/FFN/LN/residual)
     * × N rebuilds = 1280 × N tensor headers (~376 B each).
     * Reserve enough headroom for ~10k rebuilds = ~5M tensor headers.
     * The no_alloc=true ctx only holds metadata so this is cheap
     * bytes-wise. */
    s->ctx_buf_size = ggml_tensor_overhead() * 262144
                      + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false) * 4
                      + 32 * 1024 * 1024;
    s->ctx_buf = (uint8_t *)calloc(1, s->ctx_buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ s->ctx_buf_size,
        /*.mem_buffer =*/ s->ctx_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx = ggml_init(params);
    /* Graph node-count budget. Default GGML_DEFAULT_GRAPH_SIZE=2048
     * is enough for distilgpt2 (6 layers, ~1200 nodes/step) but not
     * for gpt2-small (12 layers, ~2500) and larger. 16384 covers up
     * to gpt2-xl (48 layers). Cost is one int slot per node header. */
    s->graph   = ggml_new_graph_custom(s->ctx, 16384, false);
    s->graph_b = ggml_new_graph_custom(s->ctx, 16384, false);

    /* Weights ctx pool. Sized for ~1024 weight tensors -- generous
     * upper bound that covers FullForwardFFICache at LLM scale
     * (per layer: 2 norms + 3*n_heads + 3 = up to ~50 tensors; for
     * 16 layers that's 800; plus global). no_alloc=true so this is
     * just metadata bytes. */
    /* Persistent-weights ctx. One slot per tensor declared via
     * tnn_input_*_f32_persistent. GPT-2 sizes:
     *   distilgpt2  6 layers  ~  636 tensors
     *   gpt2-small 12 layers  ~ 1272 tensors
     *   gpt2-large 36 layers  ~ 7560 tensors
     *   gpt2-xl    48 layers  ~10080 tensors  (KV cache per head adds)
     * 16384 covers up to gpt2-xl comfortably; the no_alloc ctx only
     * holds metadata so the extra bytes cost nothing on small models. */
    s->ctx_w_buf_size = ggml_tensor_overhead() * 16384;
    s->ctx_w_buf = (uint8_t *)calloc(1, s->ctx_w_buf_size);
    struct ggml_init_params w_params = {
        /*.mem_size   =*/ s->ctx_w_buf_size,
        /*.mem_buffer =*/ s->ctx_w_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx_w = ggml_init(w_params);

    s->scratch = (float *)calloc(1, TNN_SCRATCH_BYTES);
    s->realized          = 0;
    s->realized_b        = 0;
    s->weights_finalized = 0;
    s->weights_buf       = NULL;
    s->last_graph        = 0;
    return (void *)s;
}

void tnn_session_free(void *sess)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_buf) ggml_backend_buffer_free(s->weights_buf);
    if (s->ctx)   ggml_free(s->ctx);
    if (s->ctx_w) ggml_free(s->ctx_w);
    free(s->ctx_buf);
    free(s->ctx_w_buf);
    free(s->scratch);
    free(s);
    /* Engine + sched are cached globally; do not free here. */
}

const char *tnn_backend_name(void *sess)
{
    if (!sess) return "(null)";
    return ((tnn_session *)sess)->engine->backend_name;
}

int tnn_link_check(void) { return 73; }

void *tnn_input_2d_f32(void *sess, int rows, int cols)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    (void)s;   /* future: validate ctx hasn't been realized */
    return (void *)ggml_new_tensor_2d(((tnn_session *)sess)->ctx, GGML_TYPE_F32,
                                       (int64_t)cols, (int64_t)rows);
}

/* Create a PERSISTENT 2D F32 tensor in ctx_w. Its backend buffer is
 * allocated by tnn_finalize_weights (call once after all persistent
 * tensors are declared) and retained across sched_reset cycles, so
 * uploaded data survives multiple compute calls without re-upload. */
void *tnn_input_2d_f32_persistent(void *sess, int rows, int cols)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    return (void *)ggml_new_tensor_2d(s->ctx_w, GGML_TYPE_F32,
                                       (int64_t)cols, (int64_t)rows);
}

/* Same as above but 1D — used for the 7-elem adamw_params vector. */
void *tnn_input_1d_f32_persistent(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    return (void *)ggml_new_tensor_1d(s->ctx_w, GGML_TYPE_F32, (int64_t)n);
}

/* Allocate the backend buffer for all persistent tensors in ctx_w.
 * Must be called AFTER declaring all persistent tensors and BEFORE
 * any tnn_realize/compute. After this, the persistent tensors have
 * stable backend storage independent of sched.
 *
 * Returns 0 on success, negative on failure. */
int tnn_finalize_weights(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return -2;
    s->weights_buf = ggml_backend_alloc_ctx_tensors(s->ctx_w, s->engine->backend);
    if (!s->weights_buf) return -3;
    s->weights_finalized = 1;
    return 0;
}

void *tnn_matmul(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_mul_mat(s->ctx,
                                 (struct ggml_tensor *)a,
                                 (struct ggml_tensor *)b);
}

void *tnn_matmul_axb(void *sess, void *a, void *b)
{
    /* Compute A · B (no transpose at the caller). ggml_mul_mat does
     * A · B^T natively, so we transpose B first.  ggml_transpose is a
     * stride-permutation view; ggml_cont materializes it as contiguous
     * so mul_mat's contiguity-required input is satisfied. */
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *bT = ggml_cont(s->ctx, ggml_transpose(s->ctx, (struct ggml_tensor *)b));
    return (void *)ggml_mul_mat(s->ctx, (struct ggml_tensor *)a, bT);
}

void *tnn_add(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_add(s->ctx,
                             (struct ggml_tensor *)a,
                             (struct ggml_tensor *)b);
}

void *tnn_gelu(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_gelu uses the tanh approximation:
     *   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     * which matches the project's feed_forward GeLU exactly. */
    return (void *)ggml_gelu(s->ctx, (struct ggml_tensor *)a);
}

void *tnn_rms_norm(void *sess, void *x, void *gamma_row, double eps)
{
    if (!sess || !x || !gamma_row) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_rms_norm normalizes along ne[0] (the feature dim). The result
     * is the unscaled normalized tensor; we then multiply by gamma_row
     * (shape 1 x feature) which ggml_mul broadcasts over the leading dim. */
    struct ggml_tensor *normed = ggml_rms_norm(s->ctx,
                                                (struct ggml_tensor *)x,
                                                (float)eps);
    return (void *)ggml_mul(s->ctx, normed, (struct ggml_tensor *)gamma_row);
}

/* LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta. ggml_norm
 * computes the normalized (x - mean)/sqrt(var+eps) part; we then
 * multiply by gamma and add beta. Used for HF-style models (GPT-2 /
 * GPT-Neo / TinyStories) that use LayerNorm rather than RMSNorm. */
void *tnn_layer_norm(void *sess, void *x, void *gamma_row, void *beta_row, double eps)
{
    if (!sess || !x || !gamma_row || !beta_row) return NULL;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *normed = ggml_norm(s->ctx,
                                             (struct ggml_tensor *)x,
                                             (float)eps);
    struct ggml_tensor *scaled = ggml_mul(s->ctx, normed,
                                            (struct ggml_tensor *)gamma_row);
    return (void *)ggml_add(s->ctx, scaled,
                              (struct ggml_tensor *)beta_row);
}

/* Write `b` into `a` at byte offset, with row stride nb1. Result has
 * `a`'s shape (unlike ggml_cpy which returns the small dst view) so
 * downstream ops can read the modified `a` directly. Used for V[:, pos]
 * column writes in KV cache (V layout = [max_T, d_head], offset =
 * pos * 4, nb1 = max_T * 4). */
void *tnn_set_2d(void *sess, void *a, void *b, long nb1, long offset)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_set_2d(s->ctx,
                                 (struct ggml_tensor *)a,
                                 (struct ggml_tensor *)b,
                                 (size_t)nb1,
                                 (size_t)offset);
}

/* Write `b`'s rows into `a` at row indices `idx`. For our KV cache:
 *   a   = persistent K (ne=[d_head, max_T])
 *   b   = compute k_new (ne=[d_head, 1])
 *   idx = compute (1,) int32 holding the current decode position
 * The new k row lands at K[idx[0]] (other rows untouched). Same shape
 * pattern for V. Position is a RUNTIME tensor — the graph stays
 * static across decode steps, so we don't need to rebuild it. */
void *tnn_set_rows(void *sess, void *a, void *b, void *idx)
{
    if (!sess || !a || !b || !idx) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_set_rows(s->ctx,
                                   (struct ggml_tensor *)a,
                                   (struct ggml_tensor *)b,
                                   (struct ggml_tensor *)idx);
}

/* Softmax-with-mask. Adds `mask` to `a`, scales by `scale`, then runs
 * softmax along ne[0]. For KV-cache attention: scores shape (max_T, 1),
 * mask shape (max_T, 1), result shape (max_T, 1). The mask is uploaded
 * per step with 0.0 for positions <= pos and -inf for positions > pos
 * so the softmax zeroes out future-key attention even though K's
 * future-position slots may hold stale or uninitialised values. */
void *tnn_soft_max_ext(void *sess, void *a, void *mask, double scale, double max_bias)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_soft_max_ext(s->ctx,
                                       (struct ggml_tensor *)a,
                                       (struct ggml_tensor *)mask,
                                       (float)scale,
                                       (float)max_bias);
}

/* Returns a NULL pointer typed as :ptr. Useful as an Array<:ptr> seed
 * value so Spinel infers the array as a PtrArray rather than typing
 * it from a `[nil]` literal (which can resolve to IntArray). */
void *tnn_null_ptr(void)
{
    return NULL;
}

/* 1-D view of a tensor at byte `offset`, of length `ne0`. Used to
 * slice a single row out of a (max_T, d_head) KV buffer at a runtime
 * position computed by the caller (offset = pos * d_head * 4). */
void *tnn_view_1d(void *sess, void *a, int ne0, long offset)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_view_1d(s->ctx, (struct ggml_tensor *)a,
                                  (int64_t)ne0, (size_t)offset);
}

/* 2-D view of a tensor: rows of length ne0 stride nb1, ne1 rows
 * total, starting at byte `offset`. Used for slicing K/V[0:pos+1] in
 * attention. nb1 = d_head * 4 for our row-of-floats KV layout. */
void *tnn_view_2d(void *sess, void *a, int ne0, int ne1, long nb1, long offset)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_view_2d(s->ctx, (struct ggml_tensor *)a,
                                  (int64_t)ne0, (int64_t)ne1,
                                  (size_t)nb1, (size_t)offset);
}

/* Copy a -> b. Used to write k_new into a view of the persistent K
 * buffer (b = view_2d(K, d_head, 1, ..., offset=pos*d_head*4)). */
void *tnn_cpy(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_cpy(s->ctx, (struct ggml_tensor *)a,
                              (struct ggml_tensor *)b);
}

/* Concatenate `a` and `b` along the given dim (0 = ne[0], 1 = ne[1]).
 * Other dims must match. Used to glue per-head attention outputs into
 * a single (d_model, T) tensor by stacking d_head slices along ne0. */
void *tnn_concat(void *sess, void *a, void *b, int dim)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_concat(s->ctx,
                                (struct ggml_tensor *)a,
                                (struct ggml_tensor *)b,
                                dim);
}

/* Causal mask: sets elements ABOVE the diagonal (i.e. positions where
 * key_idx > query_idx + n_past) to -inf, so subsequent softmax zeroes
 * them. n_past = 0 gives the standard causal mask for training. For
 * KV-cache inference, n_past = current position so attention can see
 * cached keys plus the current token but not future tokens. */
void *tnn_diag_mask_inf(void *sess, void *a, int n_past)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_diag_mask_inf(s->ctx, (struct ggml_tensor *)a, n_past);
}

void *tnn_softmax(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_soft_max normalizes along ne[0]. With our convention
     * (ne0=cols, ne1=rows) this is per-row softmax, matching the
     * project's softmax_rows!. */
    return (void *)ggml_soft_max(s->ctx, (struct ggml_tensor *)a);
}

void *tnn_transpose(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_transpose is a stride-permutation view (no data movement).
     * Wrap in ggml_cont so the result is contiguous f32 and downloadable. */
    return (void *)ggml_cont(s->ctx,
                              ggml_transpose(s->ctx, (struct ggml_tensor *)a));
}

void *tnn_scale(void *sess, void *a, double scale)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_scale(s->ctx, (struct ggml_tensor *)a, (float)scale);
}

void *tnn_rms_norm_back(void *sess, void *x, void *dy, double eps)
{
    if (!sess || !x || !dy) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_rms_norm_back(s->ctx,
                                       (struct ggml_tensor *)x,
                                       (struct ggml_tensor *)dy,
                                       (float)eps);
}

void *tnn_softmax_back(void *sess, void *a, void *dy)
{
    if (!sess || !a || !dy) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* Plain softmax backward: scale=1.0, max_bias=0.0 (no ALiBi). */
    return (void *)ggml_soft_max_ext_back(s->ctx,
                                           (struct ggml_tensor *)a,
                                           (struct ggml_tensor *)dy,
                                           1.0f, 0.0f);
}

void *tnn_get_rows(void *sess, void *table, void *idx)
{
    if (!sess || !table || !idx) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_get_rows(s->ctx,
                                  (struct ggml_tensor *)table,
                                  (struct ggml_tensor *)idx);
}

void *tnn_get_rows_back(void *sess, void *d_out, void *idx, void *table_shape)
{
    if (!sess || !d_out || !idx || !table_shape) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_get_rows_back(s->ctx,
                                       (struct ggml_tensor *)d_out,
                                       (struct ggml_tensor *)idx,
                                       (struct ggml_tensor *)table_shape);
}

void *tnn_input_1d_i32(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_1d(s->ctx, GGML_TYPE_I32, (int64_t)n);
}

void tnn_gelu_back_scratch(void *sess, int n)
{
    if (!sess || n <= 0) return;
    tnn_session *s = (tnn_session *)sess;
    int max_slots = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (3 * n > max_slots) return;     /* not enough scratch */

    const float *x  = s->scratch + 0;
    const float *dh = s->scratch + n;
    float       *dx = s->scratch + 2 * n;

    const float c = 0.7978845608028654f;    /* sqrt(2/pi) */
    const float k = 0.044715f;

    for (int i = 0; i < n; ++i) {
        float xi  = x[i];
        float xi2 = xi * xi;
        float u   = c * (xi + k * xi * xi2);
        float tu  = tanhf(u);
        float sech2 = 1.0f - tu * tu;
        float dudx  = c * (1.0f + 3.0f * k * xi2);
        float dgelu = 0.5f * (1.0f + tu) + 0.5f * xi * sech2 * dudx;
        dx[i] = dh[i] * dgelu;
    }
}

void tnn_adam_step_scratch(void *sess, int n,
                            double lr, double b1, double b2, double eps,
                            double omc1, double omc2)
{
    if (!sess || n <= 0) return;
    tnn_session *s = (tnn_session *)sess;
    int max_slots = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (4 * n > max_slots) return;

    float *p = s->scratch + 0;
    const float *g = s->scratch + n;
    float *m = s->scratch + 2 * n;
    float *v = s->scratch + 3 * n;

    const float one_minus_b1 = (float)(1.0 - b1);
    const float one_minus_b2 = (float)(1.0 - b2);
    const float fb1   = (float)b1;
    const float fb2   = (float)b2;
    const float flr   = (float)lr;
    const float feps  = (float)eps;
    const float fomc1 = (float)omc1;
    const float fomc2 = (float)omc2;

    for (int i = 0; i < n; ++i) {
        float gi = g[i];
        float new_m = fb1 * m[i] + one_minus_b1 * gi;
        float new_v = fb2 * v[i] + one_minus_b2 * gi * gi;
        m[i] = new_m;
        v[i] = new_v;
        float m_hat = new_m / fomc1;
        float v_hat = new_v / fomc2;
        p[i] = p[i] - flr * m_hat / (sqrtf(v_hat) + feps);
    }
}

void tnn_set_output(void *tensor)
{
    if (!tensor) return;
    ggml_set_output((struct ggml_tensor *)tensor);
}

void tnn_set_param(void *tensor)
{
    if (!tensor) return;
    ggml_set_param((struct ggml_tensor *)tensor);
}

void *tnn_input_1d_f32(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_1d(s->ctx, GGML_TYPE_F32, (int64_t)n);
}

void *tnn_opt_step_adamw(void *sess, void *a, void *grad, void *m, void *v, void *params)
{
    if (!sess || !a || !grad || !m || !v || !params) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_opt_step_adamw(s->ctx,
                                        (struct ggml_tensor *)a,
                                        (struct ggml_tensor *)grad,
                                        (struct ggml_tensor *)m,
                                        (struct ggml_tensor *)v,
                                        (struct ggml_tensor *)params);
}

int tnn_realize(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)result);
    ggml_backend_sched_reset(s->engine->sched);
    if (!ggml_backend_sched_alloc_graph(s->engine->sched, s->graph)) return -3;
    s->realized   = 1;
    s->last_graph = 1;
    return 0;
}

/* Add an extra tensor's compute tree to the graph BEFORE tnn_realize.
 * Use for side-effect ops (ggml_cpy into a view) that aren't reachable
 * from the final result tensor — without this they'd be pruned. The
 * realize-target's tree is appended later by tnn_realize itself. */
int tnn_add_to_graph(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)tensor);
    return 0;
}

/* Reset for rebuild: free the compute ctx entirely and start fresh.
 * The persistent ctx_w + its backend buffer are untouched, so weights
 * keep their data. Previously this only swapped graphs in the same
 * ctx — that grew monotonically and overflowed after ~80 decode steps
 * at gpt2-small + max_T=1024 (each step creates ~1300 new tensor
 * headers, none get reclaimed). Tearing ctx down per step makes the
 * per-decode-step compute fully bounded in metadata footprint.
 *
 * The scheduler also has internal state tied to tensor pointers; we
 * reset it before realize, so this is safe. Per decode step:
 *   tnn_reset_for_rebuild(sess)
 *   ... build ops with current pos baked in ...
 *   tnn_realize(sess, result_tensor)
 *   ... upload, compute, download ... */
int tnn_reset_for_rebuild(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    /* Profile timing showed that free()+init() of the (now 130-ish MB)
     * ctx_buf adds ~500 ms per call — dominates compute. So we ONLY
     * teardown when the ctx is approaching capacity. The (small)
     * accumulated dead headers between teardowns are bounded by
     * ctx_used / ctx_buf_size, which we check before each rebuild
     * via ggml_used_mem.
     *
     * Threshold: half the buffer. Headroom ensures the *next* step's
     * graph build can complete without overflowing. */
    size_t used = ggml_used_mem(s->ctx);
    if (used > s->ctx_buf_size / 2) {
        ggml_free(s->ctx);
        struct ggml_init_params params = {
            /*.mem_size   =*/ s->ctx_buf_size,
            /*.mem_buffer =*/ s->ctx_buf,
            /*.no_alloc   =*/ true,
        };
        s->ctx        = ggml_init(params);
        s->graph_b    = ggml_new_graph_custom(s->ctx, 16384, false);
        s->realized_b = 0;
    }
    s->realized = 0;
    s->graph    = ggml_new_graph_custom(s->ctx, 16384, false);
    s->last_graph = 0;
    return 0;
}

int tnn_compute(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->engine->sched, s->graph);
    return (rc == GGML_STATUS_SUCCESS) ? 0 : (int)rc;
}

/* Build a SECONDARY graph (graph_b) in the same session, sharing ctx
 * and tensors with the primary. Does NOT alloc — call tnn_switch_b
 * before tnn_compute_b each cycle. */
int tnn_realize_b(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized_b) return -2;
    ggml_build_forward_expand(s->graph_b, (struct ggml_tensor *)result);
    s->realized_b = 1;
    return 0;
}

/* Switch sched allocation to graph_b (or back to graph). Resets the
 * scheduler then allocates buffer slots for the requested graph's
 * compute tensors. Persistent tensors (allocated via ctx_w) keep
 * their stable buffer locations. Compute tensors (h, intermediates)
 * get fresh slots that may differ from prior cycles -- caller MUST
 * re-upload any compute inputs before tnn_compute*. */
int tnn_switch_b(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized_b) return -2;
    ggml_backend_sched_reset(s->engine->sched);
    if (!ggml_backend_sched_alloc_graph(s->engine->sched, s->graph_b)) return -3;
    s->last_graph = 2;
    return 0;
}

int tnn_switch_a(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;
    ggml_backend_sched_reset(s->engine->sched);
    if (!ggml_backend_sched_alloc_graph(s->engine->sched, s->graph)) return -3;
    s->last_graph = 1;
    return 0;
}

int tnn_compute_b(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized_b) return -2;
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->engine->sched, s->graph_b);
    return (rc == GGML_STATUS_SUCCESS) ? 0 : (int)rc;
}

void tnn_scratch_set(void *sess, int idx, double v)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (idx < 0 || idx >= max_n) return;
    s->scratch[idx] = (float)v;
}

double tnn_scratch_get(void *sess, int idx)
{
    if (!sess) return 0.0;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (idx < 0 || idx >= max_n) return 0.0;
    return (double)s->scratch[idx];
}

/* The scratch buffer is just bytes; we let i32 values share it. Caller
 * must not mix i32 + f32 writes within a single tensor's upload window. */
void tnn_scratch_set_i32(void *sess, int idx, int value)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(int32_t);
    if (idx < 0 || idx >= max_n) return;
    ((int32_t *)s->scratch)[idx] = (int32_t)value;
}

int tnn_scratch_get_i32(void *sess, int idx)
{
    if (!sess) return 0;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(int32_t);
    if (idx < 0 || idx >= max_n) return 0;
    return (int)((int32_t *)s->scratch)[idx];
}

int tnn_upload(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    ggml_backend_tensor_set(t, s->scratch, 0, ggml_nbytes(t));
    return 0;
}

int tnn_download(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    ggml_backend_tensor_get(t, s->scratch, 0, ggml_nbytes(t));
    return 0;
}

int tnn_upload_from_float_array(void *sess, void *tensor, const double *data, size_t n)
{
    if (!sess || !tensor || !data) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    const size_t chunk_floats = TNN_SCRATCH_BYTES / sizeof(float);

    /* Chunked f64 → f32 conversion into scratch, then ggml_backend_tensor_set
     * per chunk at the right byte offset. Lets us upload tensors larger
     * than scratch (e.g. distilgpt2's 38.6 M-element token_embd) without
     * growing the scratch buffer for everyone. */
    size_t off = 0;
    while (off < n) {
        size_t this_chunk = (n - off) < chunk_floats ? (n - off) : chunk_floats;
        for (size_t i = 0; i < this_chunk; ++i) {
            s->scratch[i] = (float)data[off + i];
        }
        ggml_backend_tensor_set(t, s->scratch,
                                  off * sizeof(float),
                                  this_chunk * sizeof(float));
        off += this_chunk;
    }
    return 0;
}

int tnn_upload_from_int_array(void *sess, void *tensor, const long *data, size_t n)
{
    if (!sess || !tensor || !data) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    size_t max_n = TNN_SCRATCH_BYTES / sizeof(int32_t);
    if (n > max_n) return -2;

    int32_t *dst = (int32_t *)s->scratch;
    /* i64 → i32 narrowing. Spinel's :int_array is `const int64_t *`; ggml's
     * GGML_TYPE_I32 row-index tensors are 32-bit. Caller responsibility
     * not to pass out-of-range indices (vocab fits easily in int32). */
    for (size_t i = 0; i < n; ++i) dst[i] = (int32_t)data[i];

    ggml_backend_tensor_set(t, dst, 0, n * sizeof(int32_t));
    return 0;
}

int tnn_tensor_ne0(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[0] : 0; }
int tnn_tensor_ne1(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[1] : 0; }
size_t tnn_tensor_nbytes(void *t) { return t ? ggml_nbytes((struct ggml_tensor *)t) : 0; }
int    tnn_tensor_nelements(void *t) { return t ? (int)ggml_nelements((struct ggml_tensor *)t) : 0; }
