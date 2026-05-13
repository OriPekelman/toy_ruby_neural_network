#include "tinynn_ggml.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TINYNN_HAVE_CUDA
#include "ggml-cuda.h"
#endif

#include <stdlib.h>
#include <string.h>

#define TNN_SCRATCH_BYTES (16 * 1024 * 1024)   /* 16 MiB: 4M f32 */

typedef struct {
    ggml_backend_t       backend;        /* CUDA or CPU */
    ggml_backend_t       cpu_backend;    /* sched fallback when primary is CUDA */
    ggml_backend_sched_t sched;
    struct ggml_context *ctx;            /* metadata only, no_alloc=true */
    struct ggml_cgraph  *graph;
    uint8_t             *ctx_buf;
    size_t               ctx_buf_size;
    float               *scratch;
    int                  realized;
    const char          *backend_name;
} tnn_session;

void *tnn_session_new(int prefer_cuda)
{
    tnn_session *s = (tnn_session *)calloc(1, sizeof(tnn_session));
    if (!s) return NULL;

    ggml_backend_load_all();

#ifdef TINYNN_HAVE_CUDA
    if (prefer_cuda) {
        s->backend = ggml_backend_cuda_init(0);
        s->backend_name = "cuda";
    }
#endif
    if (!s->backend) {
        s->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        s->backend_name = "cpu";
    }
    if (!s->backend) { free(s); return NULL; }

    s->cpu_backend = (s->backend_name[0] == 'c' && s->backend_name[1] == 'p')
        ? NULL
        : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = s->backend;
    if (s->cpu_backend) backends[n_backends++] = s->cpu_backend;
    s->sched = ggml_backend_sched_new(backends, NULL, n_backends,
                                       GGML_DEFAULT_GRAPH_SIZE, false, true);

    s->ctx_buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
                      + ggml_graph_overhead();
    s->ctx_buf = (uint8_t *)calloc(1, s->ctx_buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ s->ctx_buf_size,
        /*.mem_buffer =*/ s->ctx_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx = ggml_init(params);
    s->graph = ggml_new_graph(s->ctx);

    s->scratch = (float *)calloc(1, TNN_SCRATCH_BYTES);
    s->realized = 0;
    return (void *)s;
}

void tnn_session_free(void *sess)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    if (s->ctx)         ggml_free(s->ctx);
    if (s->sched)       ggml_backend_sched_free(s->sched);
    if (s->cpu_backend) ggml_backend_free(s->cpu_backend);
    if (s->backend)     ggml_backend_free(s->backend);
    free(s->ctx_buf);
    free(s->scratch);
    free(s);
}

const char *tnn_backend_name(void *sess)
{
    if (!sess) return "(null)";
    return ((tnn_session *)sess)->backend_name;
}

int tnn_link_check(void) { return 73; }

void *tnn_input_2d_f32(void *sess, int rows, int cols)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_2d(s->ctx, GGML_TYPE_F32,
                                       (int64_t)cols, (int64_t)rows);
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

int tnn_realize(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)result);
    ggml_backend_sched_reset(s->sched);
    if (!ggml_backend_sched_alloc_graph(s->sched, s->graph)) return -3;
    s->realized = 1;
    return 0;
}

int tnn_compute(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->sched, s->graph);
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

int tnn_tensor_ne0(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[0] : 0; }
int tnn_tensor_ne1(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[1] : 0; }
size_t tnn_tensor_nbytes(void *t) { return t ? ggml_nbytes((struct ggml_tensor *)t) : 0; }
int    tnn_tensor_nelements(void *t) { return t ? (int)ggml_nelements((struct ggml_tensor *)t) : 0; }
