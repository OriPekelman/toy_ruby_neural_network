#ifndef TINYNN_GGUF_H
#define TINYNN_GGUF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal GGUF loading wrapper around ggml's gguf_init_from_file —
 * needed because Spinel's FFI can't pass `struct gguf_init_params` by
 * value. The session owns both the gguf metadata context and the
 * ggml context that holds tensor headers + data.
 */
void  *tnn_gguf_load(const char *path);            /* NULL on error */
void  *tnn_gguf_load_empty(void);                   /* empty GGUF, 0 tensors */
void   tnn_gguf_free(void *handle);

int    tnn_gguf_n_tensors(void *handle);
const char *tnn_gguf_tensor_name(void *handle, int i);
int    tnn_gguf_tensor_ne(void *handle, int i, int dim);   /* dim 0..3 */
int    tnn_gguf_tensor_type(void *handle, int i);           /* GGML_TYPE_F32, etc. */
size_t tnn_gguf_tensor_nbytes(void *handle, int i);

/* Copy tensor data into a caller-owned double buffer (f64).
 * `n` is the number of elements (caller should query nbytes/elementsize).
 * Currently supports GGML_TYPE_F32 only — future: dequantize Q4/Q8/etc.
 */
/* Read tensor data as doubles. Handles F32 directly and Q8_0 / Q4_K /
 * Q5_K / Q6_K / F16 / BF16 / etc. via ggml's type_traits.to_float
 * dequantization path. */
int    tnn_gguf_read_f32_to_doubles(void *handle, int i, double *out, size_t n);
int    tnn_gguf_tensor_is_quantized(void *handle, int i);

/* Read scalar metadata kv pairs. The architecture decides the keys
 * (e.g. "gpt2.embedding_length", "gpt2.attention.layer_norm_epsilon").
 * Returns -1 / 0.0 if the key is missing or has the wrong type. */
int    tnn_gguf_get_u32(void *handle, const char *key);
double tnn_gguf_get_f32(void *handle, const char *key);

/* Create a tiny GGUF file at `path` with one 2x3 f32 tensor named
 * "demo.tensor" containing [1.0, 2.0, ..., 6.0]. Useful for the
 * smoke test to exercise the full load path without needing a
 * downloaded model file. Returns 0 on success. */
int    tnn_gguf_write_demo_file(const char *path);

#ifdef __cplusplus
}
#endif

#endif
