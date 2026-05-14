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

/* Persistent F32 inputs. Created in ctx_w; backend buffer allocated by
 * tnn_finalize_weights. Data uploaded to these tensors survives across
 * sched_reset cycles — use for parameters, optimizer moments, anything
 * that should live on the device between training steps. */
void  *tnn_input_2d_f32_persistent(void *sess, int rows, int cols);
void  *tnn_input_1d_f32_persistent(void *sess, int n);

/* Allocate the backend buffer for all persistent tensors. Call once,
 * after declaring persistent tensors and before any tnn_realize. */
int    tnn_finalize_weights(void *sess);

/* Build a SECONDARY graph (graph_b) sharing the session's ctx and
 * tensors. Used for adam_step / update passes that mutate persistent
 * weights between forward calls. Does NOT alloc -- call tnn_switch_b
 * before tnn_compute_b each iteration. */
int    tnn_realize_b(void *sess, void *result);

/* Reset sched and alloc the requested graph for the next compute.
 * Persistent tensors keep their stable buffer; compute tensors get
 * fresh slots (caller must re-upload compute inputs after switch). */
int    tnn_switch_a(void *sess);
int    tnn_switch_b(void *sess);

/* Compute the secondary graph (graph_b). Must be preceded by tnn_switch_b. */
int    tnn_compute_b(void *sess);
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

/* Adam optimizer step (matches the project's adam_step_mat).
 * Scratch layout: [0..n) param, [n..2n) grad, [2n..3n) m, [3n..4n) v.
 * Updates all four in place:
 *   m_new = b1*m + (1-b1)*g
 *   v_new = b2*v + (1-b2)*g*g
 *   m_hat = m_new / omc1, v_hat = v_new / omc2
 *   param -= lr * m_hat / (sqrt(v_hat) + eps)
 * After return, read back scratch[0..n) (param), [2n..3n) (m), [3n..4n) (v).
 */
void   tnn_adam_step_scratch(void *sess, int n,
                              double lr, double b1, double b2, double eps,
                              double omc1, double omc2);

/* Realize the graph (allocates all tensors on the backend). Must be
 * called once after all ops are declared and before any upload. */
int    tnn_realize(void *sess, void *result);

/* Mark a tensor as a graph output so the backend scheduler keeps its
 * buffer alive after compute (otherwise intermediate buffers may be
 * reused as scratch by later ops, in particular ggml_gelu can reuse
 * its input tensor's allocation). Call BEFORE tnn_realize. */
void   tnn_set_output(void *tensor);

/* Mark a tensor as a trainable parameter. Required by
 * ggml_opt_step_adamw on its weight argument. Call BEFORE tnn_realize. */
void   tnn_set_param(void *tensor);

/* 1-D F32 input tensor (length n). Used for the 7-element
 * adamw_params vector (alpha, b1, b2, eps, wd, beta1h, beta2h). */
void  *tnn_input_1d_f32(void *sess, int n);

/* In-place AdamW step. After compute:
 *   m = m*b1 + g*(1-b1)
 *   v = v*b2 + g*g*(1-b2)
 *   a = a*(1 - alpha*wd) - alpha * (m*beta1h) / (sqrt(v*beta2h) + eps)
 *
 * Matches the project's plain-Adam with wd=0 since `keep = 1 - alpha*0 = 1`
 * and beta1h = 1 / (1 - b1^t), beta2h = 1 / (1 - b2^t) -- so
 * mh = m * beta1h = m_hat, vh = sqrt(v * beta2h) + eps = sqrt(v_hat) + eps. */
void  *tnn_opt_step_adamw(void *sess, void *a, void *grad, void *m, void *v, void *params);

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

/* Bulk f64 → tensor upload using Spinel's :float_array spec (matz/spinel#474).
 * `data` is a `const double *` straight from an Array<Float>'s storage; we
 * convert to f32 and bulk-copy to the backend buffer in one go. Replaces
 * the per-element tnn_scratch_set loop for row-major uploads. */
int    tnn_upload_from_float_array(void *sess, void *tensor, const double *data, size_t n);

/* Bulk i64 → tensor upload, for row-index tensors (embedding lookup).
 * The :int_array spec gives us `const int64_t *`; we narrow to int32 (which
 * is what ggml's GGML_TYPE_I32 expects) during the copy. */
int    tnn_upload_from_int_array(void *sess, void *tensor, const long *data, size_t n);

int    tnn_tensor_ne0(void *t);
int    tnn_tensor_ne1(void *t);
size_t tnn_tensor_nbytes(void *t);
int    tnn_tensor_nelements(void *t);

#ifdef __cplusplus
}
#endif

#endif
