# toy_ruby_neural_network build system.
#
# Default targets compile the Ruby drivers via Spinel. CUDA / ggml
# acceleration is opt-in:
#
#   make                   # train_minimal + train_tinystories, pure Spinel
#   make setup-ggml        # one-time clone + CPU build of vendored ggml
#   make setup-ggml-cuda   # one-time clone + CUDA build (needs CUDA toolkit)
#   make smoke             # tinynn FFI smoke test (4x3 ggml matmul demo)
#
# Vendored ggml lives at vendor/ggml/ (gitignored).
# The CUDA build expects sm_121 (NVIDIA GB10); override with
# GGML_CUDA_ARCH=NN on the command line.

SPINEL_DIR  ?= $(HOME)/sites/spinel
SPINEL      ?= $(SPINEL_DIR)/spinel

CC          ?= cc
CFLAGS      ?= -O2 -fPIC -Wall -Wextra
ARFLAGS      = rcs

# --- vendored ggml ----------------------------------------------------------
GGML_DIR    := vendor/ggml
GGML_REPO   := https://github.com/ggml-org/ggml.git
GGML_CUDA_ARCH ?= 121
CUDA_DIR    ?= /usr/local/cuda

# --- pure-Spinel drivers ----------------------------------------------------
all: train_minimal train_tinystories

train_minimal: train_minimal.rb lib/transformer.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@


train_tinystories: train_tinystories.rb lib/transformer.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# --- ggml vendor ------------------------------------------------------------
$(GGML_DIR)/CMakeLists.txt:
	mkdir -p vendor
	git clone --depth 1 $(GGML_REPO) $(GGML_DIR)

setup-ggml: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && cmake -B build \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cd $(GGML_DIR) && cmake --build build -j$(shell nproc)

setup-ggml-cuda: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH cmake -B build-cuda \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$(GGML_CUDA_ARCH) -DGGML_NATIVE=OFF
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH cmake --build build-cuda -j$(shell nproc)

# --- tinynn shim (CPU build) ------------------------------------------------
GGML_INC := -I$(GGML_DIR)/include -I$(GGML_DIR)/src

tinynn/tinynn_ggml.o: tinynn/tinynn_ggml.c tinynn/tinynn_ggml.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/tinynn_gguf.o: tinynn/tinynn_gguf.c tinynn/tinynn_gguf.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/libtinynn_ggml.a: tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o
	ar $(ARFLAGS) $@ tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o

# --- smoke test -------------------------------------------------------------
# Builds tinynn/smoke.rb against the CPU shim. Requires `setup-ggml` to have
# been run once first.
smoke: tinynn/smoke
	./tinynn/smoke

tinynn/smoke: tinynn/smoke.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/smoke.rb -o tinynn/smoke

# A/B parity tests: native vs FFI (CPU) for one op each.
ab-smoke: tinynn/ab_smoke
	./tinynn/ab_smoke

ab-smoke-add: tinynn/ab_smoke_add
	./tinynn/ab_smoke_add

ab-smoke-gelu: tinynn/ab_smoke_gelu
	./tinynn/ab_smoke_gelu

ab-smoke-rms-norm: tinynn/ab_smoke_rms_norm
	./tinynn/ab_smoke_rms_norm

ab-smoke-softmax: tinynn/ab_smoke_softmax
	./tinynn/ab_smoke_softmax

ab-smoke-transpose: tinynn/ab_smoke_transpose
	./tinynn/ab_smoke_transpose

ab-smoke-scale: tinynn/ab_smoke_scale
	./tinynn/ab_smoke_scale

# Chained-op pipeline: gelu(h·w1)·w2 in one ggml graph.
ab-smoke-pipeline: tinynn/ab_smoke_pipeline
	./tinynn/ab_smoke_pipeline

# Run every CPU smoke. (CUDA variants would need `make setup-ggml-cuda` first.)
# `ab-smoke-transpose` is omitted: ggml_cont(ggml_transpose(...)) trips
# the scheduler's buffer allocation; we fold transposes into consuming
# ops instead (see TinyNN.matmul's b-transposed upload).
test: smoke ab-smoke ab-smoke-add ab-smoke-gelu ab-smoke-rms-norm \
       ab-smoke-softmax ab-smoke-scale ab-smoke-pipeline \
       ab-smoke-matmul-variants ab-smoke-back ab-smoke-embed ab-smoke-sgd \
       ab-smoke-gelu-back ab-smoke-cegrad ab-smoke-adam

tinynn/ab_smoke: tinynn/ab_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke.rb -o tinynn/ab_smoke

tinynn/ab_smoke_add: tinynn/ab_smoke_add.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_add.rb -o tinynn/ab_smoke_add

tinynn/ab_smoke_gelu: tinynn/ab_smoke_gelu.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu.rb -o tinynn/ab_smoke_gelu

tinynn/ab_smoke_rms_norm: tinynn/ab_smoke_rms_norm.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_rms_norm.rb -o tinynn/ab_smoke_rms_norm

tinynn/ab_smoke_softmax: tinynn/ab_smoke_softmax.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_softmax.rb -o tinynn/ab_smoke_softmax

tinynn/ab_smoke_transpose: tinynn/ab_smoke_transpose.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_transpose.rb -o tinynn/ab_smoke_transpose

tinynn/ab_smoke_scale: tinynn/ab_smoke_scale.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_scale.rb -o tinynn/ab_smoke_scale

tinynn/ab_smoke_pipeline: tinynn/ab_smoke_pipeline.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_pipeline.rb -o tinynn/ab_smoke_pipeline

# Transformer-shape sized parity + wallclock comparison.
ab-smoke-big: tinynn/ab_smoke_big
	./tinynn/ab_smoke_big

tinynn/ab_smoke_big: tinynn/ab_smoke_big.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_big.rb -o tinynn/ab_smoke_big

ab-smoke-matmul-variants: tinynn/ab_smoke_matmul_variants
	./tinynn/ab_smoke_matmul_variants

tinynn/ab_smoke_matmul_variants: tinynn/ab_smoke_matmul_variants.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_matmul_variants.rb -o tinynn/ab_smoke_matmul_variants

ab-smoke-back: tinynn/ab_smoke_back
	./tinynn/ab_smoke_back

tinynn/ab_smoke_back: tinynn/ab_smoke_back.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_back.rb -o tinynn/ab_smoke_back

ab-smoke-gelu-back: tinynn/ab_smoke_gelu_back
	./tinynn/ab_smoke_gelu_back

tinynn/ab_smoke_gelu_back: tinynn/ab_smoke_gelu_back.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu_back.rb -o tinynn/ab_smoke_gelu_back

ab-smoke-cegrad: tinynn/ab_smoke_cegrad
	./tinynn/ab_smoke_cegrad

tinynn/ab_smoke_cegrad: tinynn/ab_smoke_cegrad.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cegrad.rb -o tinynn/ab_smoke_cegrad

ab-smoke-adam: tinynn/ab_smoke_adam
	./tinynn/ab_smoke_adam

tinynn/ab_smoke_adam: tinynn/ab_smoke_adam.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adam.rb -o tinynn/ab_smoke_adam

gguf-smoke: tinynn/gguf_smoke
	./tinynn/gguf_smoke

tinynn/gguf_smoke: tinynn/gguf_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_smoke.rb -o tinynn/gguf_smoke

ab-smoke-embed: tinynn/ab_smoke_embed
	./tinynn/ab_smoke_embed

tinynn/ab_smoke_embed: tinynn/ab_smoke_embed.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_embed.rb -o tinynn/ab_smoke_embed

ab-smoke-sgd: tinynn/ab_smoke_sgd
	./tinynn/ab_smoke_sgd

tinynn/ab_smoke_sgd: tinynn/ab_smoke_sgd.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_sgd.rb -o tinynn/ab_smoke_sgd

# Forward-only smoke: does TransformerLM#forward run at current Spinel
# master? (The #473 SIGBUS is in backward; forward might be OK.)
forward-smoke: tinynn/forward_smoke
	./tinynn/forward_smoke

tinynn/forward_smoke: tinynn/forward_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/forward_smoke.rb -o tinynn/forward_smoke

persistent-bench: tinynn/persistent_bench
	./tinynn/persistent_bench

tinynn/persistent_bench: tinynn/persistent_bench.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench.rb -o tinynn/persistent_bench

persistent-bench-cuda: tinynn/persistent_bench_cuda
	./tinynn/persistent_bench_cuda

tinynn/persistent_bench_cuda: tinynn/persistent_bench_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/persistent_bench_cuda.rb -o tinynn/persistent_bench_cuda

persistent-bench-big: tinynn/persistent_bench_big
	./tinynn/persistent_bench_big

tinynn/persistent_bench_big: tinynn/persistent_bench_big.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench_big.rb -o tinynn/persistent_bench_big

# A/B parity test against CUDA backend on the local GPU (sm_121 / GB10).
# Requires `make setup-ggml-cuda` to have produced vendor/ggml/build-cuda.
ab-smoke-cuda: tinynn/ab_smoke_cuda
	./tinynn/ab_smoke_cuda

tinynn/tinynn_ggml_cuda.o: tinynn/tinynn_ggml.c tinynn/tinynn_ggml.h
	$(CC) $(CFLAGS) -DTINYNN_HAVE_CUDA=1 $(GGML_INC) -I$(CUDA_DIR)/include -c $< -o $@

tinynn/libtinynn_ggml_cuda.a: tinynn/tinynn_ggml_cuda.o
	ar $(ARFLAGS) $@ $<

tinynn/ab_smoke_cuda: tinynn/ab_smoke_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_cuda.rb -o tinynn/ab_smoke_cuda

# Consolidated CUDA parity test: matmul + add + gelu + rms_norm + softmax + scale + ffn_pipeline.
ab-smoke-all-cuda: tinynn/ab_smoke_all_cuda
	./tinynn/ab_smoke_all_cuda

tinynn/ab_smoke_all_cuda: tinynn/ab_smoke_all_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_all_cuda.rb -o tinynn/ab_smoke_all_cuda

# Transformer-shape parity + wallclock bench on CUDA (GB10).
ab-smoke-big-cuda: tinynn/ab_smoke_big_cuda
	./tinynn/ab_smoke_big_cuda

tinynn/ab_smoke_big_cuda: tinynn/ab_smoke_big_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_big_cuda.rb -o tinynn/ab_smoke_big_cuda

# --- maintenance ------------------------------------------------------------
clean:
	rm -f train_minimal train_tinystories \
	      tinynn/tinynn_ggml.o tinynn/libtinynn_ggml.a \
	      tinynn/tinynn_ggml_cuda.o tinynn/libtinynn_ggml_cuda.a \
	      tinynn/smoke tinynn/ab_smoke tinynn/ab_smoke_cuda tinynn/ab_smoke_all_cuda \
	      tinynn/ab_smoke_add tinynn/ab_smoke_gelu tinynn/ab_smoke_rms_norm \
	      tinynn/ab_smoke_softmax tinynn/ab_smoke_transpose tinynn/ab_smoke_scale \
	      tinynn/ab_smoke_pipeline tinynn/ab_smoke_big tinynn/ab_smoke_big_cuda \
	      tinynn/ab_smoke_matmul_variants tinynn/ab_smoke_back tinynn/ab_smoke_embed \
	      tinynn/ab_smoke_sgd tinynn/ab_smoke_gelu_back tinynn/ab_smoke_cegrad \
	      tinynn/ab_smoke_adam tinynn/forward_smoke tinynn/persistent_bench \
	      tinynn/persistent_bench_cuda tinynn/persistent_bench_big

distclean: clean
	rm -rf $(GGML_DIR)/build $(GGML_DIR)/build-cuda

.PHONY: all clean distclean setup-ggml setup-ggml-cuda smoke \
        ab-smoke ab-smoke-add ab-smoke-gelu ab-smoke-rms-norm \
        ab-smoke-softmax ab-smoke-transpose ab-smoke-scale \
        ab-smoke-pipeline ab-smoke-big ab-smoke-cuda ab-smoke-all-cuda \
        ab-smoke-big-cuda test
