#include "tinynn_gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    struct gguf_context *gguf_ctx;
    struct ggml_context *ggml_ctx;
} tnn_gguf_session;

void *tnn_gguf_load(const char *path)
{
    if (!path) return NULL;
    struct ggml_context *ggml_ctx = NULL;
    struct gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    struct gguf_context *gctx = gguf_init_from_file(path, params);
    if (!gctx) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)calloc(1, sizeof(*s));
    if (!s) { gguf_free(gctx); if (ggml_ctx) ggml_free(ggml_ctx); return NULL; }
    s->gguf_ctx = gctx;
    s->ggml_ctx = ggml_ctx;
    return s;
}

void *tnn_gguf_load_empty(void)
{
    tnn_gguf_session *s = (tnn_gguf_session *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->gguf_ctx = gguf_init_empty();
    s->ggml_ctx = NULL;
    return s;
}

void tnn_gguf_free(void *handle)
{
    if (!handle) return;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (s->ggml_ctx) ggml_free(s->ggml_ctx);
    if (s->gguf_ctx) gguf_free(s->gguf_ctx);
    free(s);
}

int tnn_gguf_n_tensors(void *handle)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return (int)gguf_get_n_tensors(s->gguf_ctx);
}

const char *tnn_gguf_tensor_name(void *handle, int i)
{
    if (!handle) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
}

int tnn_gguf_tensor_ne(void *handle, int i, int dim)
{
    if (!handle || dim < 0 || dim >= 4) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (!s->ggml_ctx) return 0;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
    if (!name) return 0;
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    return t ? (int)t->ne[dim] : 0;
}

int tnn_gguf_tensor_type(void *handle, int i)
{
    if (!handle) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return (int)gguf_get_tensor_type(s->gguf_ctx, (int64_t)i);
}

size_t tnn_gguf_tensor_nbytes(void *handle, int i)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return gguf_get_tensor_size(s->gguf_ctx, (int64_t)i);
}

int tnn_gguf_read_f32_to_doubles(void *handle, int i, double *out, size_t n)
{
    if (!handle || !out) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (!s->ggml_ctx) return -2;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
    if (!name) return -3;
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    if (!t || !t->data) return -4;
    size_t available = ggml_nelements(t);
    if (n > available) n = available;

    /* F32: direct copy. */
    if (t->type == GGML_TYPE_F32) {
        const float *src = (const float *)t->data;
        for (size_t k = 0; k < n; ++k) out[k] = (double)src[k];
        return 0;
    }

    /* Quantized (Q8_0, Q4_K, Q5_K, F16, BF16, …): dequantize through
     * ggml's type_traits.to_float. Allocates a temporary f32 buffer
     * sized to the tensor's full element count (read past `n` not
     * supported per-block-aligned dequantization). */
    const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) return -5;

    /* to_float must dequantize whole blocks; we ask for `available`
     * elements (the full tensor) and then only forward `n` of them. */
    float *fbuf = (float *)malloc(available * sizeof(float));
    if (!fbuf) return -6;
    traits->to_float(t->data, fbuf, (int64_t)available);
    for (size_t k = 0; k < n; ++k) out[k] = (double)fbuf[k];
    free(fbuf);
    return 0;
}

/* Convenience: tell the caller whether a given tensor index needs
 * dequantization (i.e. is not GGML_TYPE_F32). Useful for the Ruby
 * side to print "loaded Q8_0 tensor X" type diagnostics. */
int tnn_gguf_tensor_is_quantized(void *handle, int i)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    enum ggml_type t = gguf_get_tensor_type(s->gguf_ctx, (int64_t)i);
    return (t == GGML_TYPE_F32) ? 0 : 1;
}

/* Look up a uint32 metadata key (e.g. "gpt2.embedding_length"). Returns
 * the value, or -1 if the key is missing or has a non-u32 type. The
 * caller is expected to know which kv key names are valid for the
 * file's architecture. */
int tnn_gguf_get_u32(void *handle, const char *key)
{
    if (!handle || !key) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return -1;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_UINT32) return -1;
    return (int)gguf_get_val_u32(s->gguf_ctx, kid);
}

/* Same shape for float-typed metadata (e.g. layer_norm_epsilon). */
double tnn_gguf_get_f32(void *handle, const char *key)
{
    if (!handle || !key) return 0.0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return 0.0;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_FLOAT32) return 0.0;
    return (double)gguf_get_val_f32(s->gguf_ctx, kid);
}

/* Bool metadata. Returns 1 if key is present and true, 0 if false or
 * missing. */
int tnn_gguf_get_bool(void *handle, const char *key)
{
    if (!handle || !key) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return 0;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_BOOL) return 0;
    return gguf_get_val_bool(s->gguf_ctx, kid) ? 1 : 0;
}

/* --- direct GGUF → FFI persistent buffer loaders ------------------ */

/* Helper: get a pointer to the tensor's f32 data, dequantizing into
 * `*owned_buf` if the source is quantized/F16/BF16. Caller must free
 * `*owned_buf` if it's set to non-NULL on return. Returns NULL on error. */
static const float *gguf_tensor_as_f32(tnn_gguf_session *s, int tensor_idx,
                                        size_t expect_n,
                                        float **owned_buf, int *err)
{
    *owned_buf = NULL;
    *err = 0;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)tensor_idx);
    if (!name) { *err = -3; return NULL; }
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    if (!t || !t->data) { *err = -4; return NULL; }
    size_t avail = ggml_nelements(t);
    if (expect_n > 0 && expect_n != avail) {
        /* Tolerated when expect_n==0 (caller doesn't know); otherwise
         * a mismatch likely means the caller has the wrong shape. */
        *err = -7;
        return NULL;
    }
    if (t->type == GGML_TYPE_F32) {
        return (const float *)t->data;
    }
    const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) { *err = -5; return NULL; }
    float *fbuf = (float *)malloc(avail * sizeof(float));
    if (!fbuf) { *err = -6; return NULL; }
    traits->to_float(t->data, fbuf, (int64_t)avail);
    *owned_buf = fbuf;
    return fbuf;
}

int tnn_gguf_find_index(void *handle, const char *name)
{
    if (!handle || !name) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int n = gguf_get_n_tensors(s->gguf_ctx);
    for (int i = 0; i < n; ++i) {
        const char *cur = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
        if (cur && strcmp(cur, name) == 0) return i;
    }
    return -1;
}

int tnn_gguf_copy_to_persistent(void *handle, int tensor_idx,
                                 void *sess, void *target_tensor)
{
    (void)sess;  /* unused: ggml_backend_tensor_set works without scratch */
    if (!handle || !target_tensor || tensor_idx < 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;
    size_t dst_n = ggml_nelements(dst);

    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, dst_n, &owned, &err);
    if (!src) return err;

    ggml_backend_tensor_set(dst, src, 0, dst_n * sizeof(float));
    free(owned);
    return 0;
}

int tnn_gguf_copy_1d_to_persistent(void *handle, int tensor_idx,
                                    void *sess, void *target_tensor)
{
    /* 1-D direct copy — same wire as 2-D for our purposes. */
    return tnn_gguf_copy_to_persistent(handle, tensor_idx, sess, target_tensor);
}

/* Chunked transposed copy. Source is (br × bc) row-major; destination
 * is a ggml tensor with ne=[br, bc] whose backend storage we fill with
 * the transposed byte order. Same shape as tnn_upload_transposed_f64. */
int tnn_gguf_copy_transposed_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int br, int bc)
{
    (void)sess;
    if (!handle || !target_tensor || br <= 0 || bc <= 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;
    size_t expected = (size_t)br * (size_t)bc;
    if (expected * sizeof(float) > ggml_nbytes(dst)) return -2;

    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, expected, &owned, &err);
    if (!src) return err;

    /* Stage one chunk of columns at a time into a single allocated
     * buffer (sized to hold the chunk), then ggml_backend_tensor_set
     * with byte offset. We allocate the staging buffer once per call
     * — 16 MiB max, same as the session scratch — and reuse it
     * across iterations of the column loop. */
    const size_t chunk_floats_max = 4 * 1024 * 1024;   /* 16 MiB */
    int cols_per_chunk = (int)(chunk_floats_max / (size_t)br);
    if (cols_per_chunk <= 0) {
        if (owned) free(owned);
        return -3;
    }
    size_t stage_bytes = (size_t)cols_per_chunk * (size_t)br * sizeof(float);
    float *stage = (float *)malloc(stage_bytes);
    if (!stage) {
        if (owned) free(owned);
        return -6;
    }

    int j_start = 0;
    while (j_start < bc) {
        int j_end = j_start + cols_per_chunk;
        if (j_end > bc) j_end = bc;
        int j = j_start;
        while (j < j_end) {
            const float *src_col_strided = src + j;
            float *dst_col = stage + (size_t)(j - j_start) * (size_t)br;
            int i = 0;
            while (i < br) {
                dst_col[i] = src_col_strided[(size_t)i * (size_t)bc];
                i++;
            }
            j++;
        }
        size_t byte_off = (size_t)j_start * (size_t)br * sizeof(float);
        size_t byte_len = (size_t)(j_end - j_start) * (size_t)br * sizeof(float);
        ggml_backend_tensor_set(dst, stage, byte_off, byte_len);
        j_start = j_end;
    }
    free(stage);
    free(owned);
    return 0;
}

/* Extract one head-slice of a [d_model, n_heads_total * d_head] tensor
 * and transpose-write it into a ne=[d_model, d_head] persistent target.
 *
 * Source row stride = n_heads_total * d_head; head h's columns are
 * [h*d_head, (h+1)*d_head). Per-head slice is (d_model × d_head)
 * row-major. We transpose-stage that into a chunked buffer, same
 * layout convention as tnn_upload_transposed_f64. */
int tnn_gguf_copy_head_slice_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int head_idx, int n_heads_total,
                                            int d_model, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || n_heads_total <= 0
        || d_model <= 0 || d_head <= 0) return -1;
    if (head_idx >= n_heads_total) return -1;

    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t total = (size_t)d_model * (size_t)n_heads_total * (size_t)d_head;
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, total, &owned, &err);
    if (!src) return err;

    /* src layout: row i (i ∈ [0, d_model)) has n_heads_total*d_head
     * elements. Head h's d_head sub-row is at src + i*src_cols + h*d_head. */
    const int src_cols = n_heads_total * d_head;
    const int col_off  = head_idx * d_head;

    /* Stage transposed: dst ggml tensor ne=[d_model, d_head]. For
     * each "ne1 column" k (0..d_head), write d_model floats from
     * src[*, col_off + k]. Single chunk if (d_head * d_model) fits;
     * for our shapes d_model up to 4096, d_head 64..128 it always
     * fits in 16 MiB. */
    const size_t chunk_floats_max = 4 * 1024 * 1024;
    int cols_per_chunk = (int)(chunk_floats_max / (size_t)d_model);
    if (cols_per_chunk <= 0) {
        if (owned) free(owned);
        return -3;
    }
    if (cols_per_chunk > d_head) cols_per_chunk = d_head;

    size_t stage_bytes = (size_t)cols_per_chunk * (size_t)d_model * sizeof(float);
    float *stage = (float *)malloc(stage_bytes);
    if (!stage) {
        if (owned) free(owned);
        return -6;
    }

    int k_start = 0;
    while (k_start < d_head) {
        int k_end = k_start + cols_per_chunk;
        if (k_end > d_head) k_end = d_head;
        int k = k_start;
        while (k < k_end) {
            float *dst_col = stage + (size_t)(k - k_start) * (size_t)d_model;
            int i = 0;
            const float *src_row = src + col_off + k;
            while (i < d_model) {
                dst_col[i] = src_row[(size_t)i * (size_t)src_cols];
                i++;
            }
            k++;
        }
        size_t byte_off = (size_t)k_start * (size_t)d_model * sizeof(float);
        size_t byte_len = (size_t)(k_end - k_start) * (size_t)d_model * sizeof(float);
        ggml_backend_tensor_set(dst, stage, byte_off, byte_len);
        k_start = k_end;
    }
    free(stage);
    free(owned);
    return 0;
}

/* Native-layout per-head slice: HF-native source has shape
 * [n_heads_total * d_head, d_model] row-major. Head h occupies rows
 * [h*d_head, (h+1)*d_head); that block is a contiguous (d_head, d_model)
 * row-major slice. Those bytes already match ggml ne=[d_model, d_head]
 * column-major (because for j in d_head: for i in d_model: is the same
 * iteration order), so this is a plain ggml_backend_tensor_set with no
 * transpose stage. Used by GGUFLoad.load_kv_cache_directly_native to
 * fill the per-head t_w_q / t_w_k / t_w_v persistent tensors. */
int tnn_gguf_copy_head_slice_to_persistent_native(void *handle, int tensor_idx,
                                                    void *sess, void *target_tensor,
                                                    int head_idx, int n_heads_total,
                                                    int d_model, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || n_heads_total <= 0
        || d_model <= 0 || d_head <= 0) return -1;
    if (head_idx >= n_heads_total) return -1;

    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t total = (size_t)d_model * (size_t)n_heads_total * (size_t)d_head;
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, total, &owned, &err);
    if (!src) return err;

    size_t head_floats = (size_t)d_head * (size_t)d_model;
    size_t head_off    = (size_t)head_idx * head_floats;
    ggml_backend_tensor_set(dst, src + head_off, 0, head_floats * sizeof(float));
    free(owned);
    return 0;
}

/* Extract one head-slice of a 1-D bias tensor [n_heads_total*d_head]
 * and copy d_head floats into the 1-D target (length d_head). */
int tnn_gguf_copy_head_bias_slice_to_persistent(void *handle, int tensor_idx,
                                                  void *sess, void *target_tensor,
                                                  int head_idx, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || d_head <= 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    /* Don't require n_heads_total; instead read the tensor and slice
     * by byte offset based on head_idx*d_head*sizeof(float). The full
     * tensor is dequant'd (cheap — at most n_heads*d_head floats). */
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, 0, &owned, &err);
    if (!src) return err;

    const float *head_slice = src + (size_t)head_idx * (size_t)d_head;
    ggml_backend_tensor_set(dst, head_slice, 0, (size_t)d_head * sizeof(float));
    free(owned);
    return 0;
}

int tnn_gguf_write_demo_file(const char *path)
{
    if (!path) return -1;

    /* Build a tiny ggml ctx, allocate one 2x3 f32 tensor, fill it. */
    size_t mem_size = 64 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return -2;
    struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);   /* ne0=3 cols, ne1=2 rows */
    ggml_set_name(t, "demo.tensor");
    float *d = (float *)t->data;
    for (int k = 0; k < 6; ++k) d[k] = (float)(k + 1);

    struct gguf_context *gctx = gguf_init_empty();
    gguf_add_tensor(gctx, t);
    int rc = gguf_write_to_file(gctx, path, false) ? 0 : -3;

    gguf_free(gctx);
    ggml_free(ctx);
    return rc;
}
