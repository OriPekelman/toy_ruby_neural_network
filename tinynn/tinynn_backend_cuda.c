/* tinynn_backend_cuda.c — CUDA backend init + BYO-pointer hook.
 *
 * Lives in its own .o so libtinynn_ggml_cuda.a contains only the
 * symbols that bridge to ggml-cuda — no overlap with libtinynn_ggml.a.
 * The common-side tnn_engine_get / tnn_session_attach_weight_mmap (in
 * tinynn_ggml.c) call these through weak references, so CPU-only
 * programs link cleanly without this archive and CUDA programs pull
 * the needed symbols.
 *
 * Compiled only into libtinynn_ggml_cuda.a (rule in Makefile).
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

ggml_backend_t tnn_backend_cuda_init_internal(void)
{
    return ggml_backend_cuda_init(0);
}

/* Phase 2 BYO-pointer on CUDA: wraps an mmap'd region in a
 * ggml_backend_cuda_buffer_from_ptr (vendored patch — see
 * docs/cuda-byo-pointer-design.md). Strong override of the weak
 * stub in tinynn_ggml.c. */
ggml_backend_buffer_t tnn_cuda_buffer_from_ptr_internal(void *host_ptr,
                                                         size_t size,
                                                         int device)
{
    return ggml_backend_cuda_buffer_from_ptr(host_ptr, size, device);
}
