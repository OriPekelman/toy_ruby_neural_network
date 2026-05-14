#include "tinynn_gguf.h"
#include "ggml.h"
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
    if (t->type != GGML_TYPE_F32) return -5;   /* TODO: dequantize quantized types */
    size_t available = ggml_nelements(t);
    if (n > available) n = available;
    const float *src = (const float *)t->data;
    for (size_t k = 0; k < n; ++k) out[k] = (double)src[k];
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
