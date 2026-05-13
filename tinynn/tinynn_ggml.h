#ifndef TINYNN_GGML_H
#define TINYNN_GGML_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Spinel-friendly C shim over ggml. Backend-aware (CPU now; CUDA when
 * compiled with -DTINYNN_HAVE_CUDA against the ggml-cuda build).
 *
 * Tensor lifetime: tied to the session. tnn_session_free disposes all
 * tensors created via the session.
 *
 * Result-shape convention for matmul: ggml's mul_mat result is
 * (ne0=m, ne1=n) where m = rows of A and n = rows of B (A,B given as
 * ne0=k, ne1=rows). Reading the logical (m,n) row-major result: index
 * scratch[j*m + i] after tnn_tensor_pull. The Ruby side does the swap.
 */

void  *tnn_session_new(int prefer_cuda);     /* 0 = CPU only, 1 = CUDA if compiled in */
void   tnn_session_free(void *sess);

const char *tnn_backend_name(void *sess);
int    tnn_link_check(void);                 /* returns 73 */

/* Build phase: declare tensors and ops, then realize.
 * Tensors are created before tnn_realize; backend storage is allocated then. */
void  *tnn_input_2d_f32(void *sess, int rows, int cols);
void  *tnn_matmul(void *sess, void *a, void *b);        /* A * B^T (ggml-native) */
void  *tnn_matmul_axb(void *sess, void *a, void *b);    /* A * B  (transposes B internally) */
void  *tnn_add(void *sess, void *a, void *b);           /* element-wise A + B (same shape) */
void  *tnn_gelu(void *sess, void *a);                   /* element-wise GeLU (tanh approx) */
void  *tnn_rms_norm(void *sess, void *x, void *gamma_row, double eps);
                                                         /* RMSNorm(x) * gamma_row, last-dim normalize, broadcast over the leading dim.
                                                            x: (n1, n0) with ne0=feature, ne1=batch_or_T
                                                            gamma_row: (1, n0) — a 1xfeature tensor */
void  *tnn_softmax(void *sess, void *a);                /* per-row softmax along ne[0] */
void  *tnn_transpose(void *sess, void *a);              /* materialised transpose: (rows,cols) → (cols,rows) */
void  *tnn_scale(void *sess, void *a, double s);        /* element-wise a * s */

/* Backward ops. */
void  *tnn_rms_norm_back(void *sess, void *x, void *dy, double eps);
                                                         /* d/dx of plain RMSNorm(x) (no gamma); caller handles gamma chain. */
void  *tnn_softmax_back(void *sess, void *a, void *dy); /* d/dx of per-row softmax. a is softmax output. */
void  *tnn_get_rows(void *sess, void *table, void *idx);
                                                         /* gather rows: out[i] = table[idx[i]] */
void  *tnn_get_rows_back(void *sess, void *d_out, void *idx, void *table_shape);
                                                         /* scatter-add: out has table's shape, out[idx[i]] += d_out[i] */
void  *tnn_input_1d_i32(void *sess, int n);             /* int32 vector input (for row indices). */

/* Custom non-ggml kernel: dx = dh * d/dx GeLU(x) (tanh approximation).
 * Reads x from scratch[0..n) and dh from scratch[n..2n); writes
 * dx into scratch[2n..3n). Caller is responsible for staging and
 * reading back. CPU-only (no GPU path yet).
 *
 * No new tensor created; this is a side-channel op that skips the
 * ggml graph entirely. Useful because ggml has no gelu_back op.
 */
void   tnn_gelu_back_scratch(void *sess, int n);

/* Realize the graph (allocates all tensors on the backend). Must be
 * called once after all ops are declared and before any upload. */
int    tnn_realize(void *sess, void *result);

/* Compute the (already-built) graph. Must be called after upload. */
int    tnn_compute(void *sess);

/* Bulk upload/download via a session-owned scratch buffer (16 MiB).
 *   - tnn_scratch_set(sess, idx, value) fills scratch element by element
 *     (per-element FFI call; cheap on the Ruby side).
 *   - tnn_upload(sess, tensor) bulk-copies the staged data into the
 *     backend buffer (single backend_tensor_set => one cudaMemcpy).
 *   - tnn_download(sess, tensor) pulls in the other direction.
 *   - tnn_scratch_get reads from the host shadow.
 */
void   tnn_scratch_set(void *sess, int idx, double v);
double tnn_scratch_get(void *sess, int idx);
void   tnn_scratch_set_i32(void *sess, int idx, int value);
int    tnn_scratch_get_i32(void *sess, int idx);
int    tnn_upload(void *sess, void *tensor);
int    tnn_download(void *sess, void *tensor);

int    tnn_tensor_ne0(void *t);
int    tnn_tensor_ne1(void *t);
size_t tnn_tensor_nbytes(void *t);
int    tnn_tensor_nelements(void *t);

#ifdef __cplusplus
}
#endif

#endif
