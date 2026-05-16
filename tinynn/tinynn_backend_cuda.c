/* tinynn_backend_cuda.c — JUST the CUDA backend init.
 *
 * Lives in its own .o so libtinynn_ggml_cuda.a contains only this
 * symbol — no overlap with libtinynn_ggml.a. The common-side
 * tnn_engine_get (in tinynn_ggml.c) calls this through a weak
 * reference, so CPU-only programs link cleanly without this archive
 * and CUDA programs pull just the one symbol they need.
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
