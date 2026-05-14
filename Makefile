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

# macOS Command Line Tools (as of 26.x) keep stale 2023 C++ stub headers
# at /Library/Developer/CommandLineTools/usr/include/c++/v1 which shadow
# the real headers in the SDK. Prepend the SDK's libc++ include path so
# ggml's C++ files can find <mutex>, <array>, etc. No-op on Linux.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CMAKE_ENV := CPLUS_INCLUDE_PATH=$(shell xcrun --show-sdk-path)/usr/include/c++/v1
  NJOBS     := $(shell sysctl -n hw.logicalcpu)
else
  CMAKE_ENV :=
  NJOBS     := $(shell nproc)
endif

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

# Inference demo: greedy autoregressive generation via FullForwardFFICache.
# Parity-checks vs native TransformerLM.forward.
inference_demo: inference_demo.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

inference_demo_cuda: inference_demo_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

# Tep+Spinel HTTP server demos. See tep_demo/README.md. Builds bypass
# tep's translator (we use spinel directly on a layout-substituted
# copy of tep's lib in tep_demo/_tep_lib/).
tep_demo/hello: tep_demo/hello_api.rb tep_demo/_tep_lib/tep.rb
	$(SPINEL) tep_demo/hello_api.rb -o tep_demo/hello

# Inference API: /generate?n=N runs greedy generation via FullForwardFFICache.
tep_demo/api: tep_demo/inference_api.rb tep_demo/_tep_lib/tep.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/inference_api.rb -o tep_demo/api

# --- ggml vendor ------------------------------------------------------------
$(GGML_DIR)/CMakeLists.txt:
	mkdir -p vendor
	git clone --depth 1 $(GGML_REPO) $(GGML_DIR)

# GGML_OPENMP=OFF: avoid the libgomp link dependency. On macOS clang
# ships libomp (LLVM), not libgomp (GNU); ggml's own thread pool covers
# CPU parallelism either way. Same setting used on Linux for build
# parity (and so lib/tinynn.rb doesn't need ffi_lib "gomp").
setup-ggml: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && $(CMAKE_ENV) cmake -B build \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cd $(GGML_DIR) && $(CMAKE_ENV) cmake --build build -j$(NJOBS)

setup-ggml-cuda: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake -B build-cuda \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$(GGML_CUDA_ARCH) -DGGML_NATIVE=OFF
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake --build build-cuda -j$(NJOBS)

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

# Chained FFNFFICache parity: pre, hidden, out vs hand-rolled native.
ab-smoke-ffncache: tinynn/ab_smoke_ffncache
	./tinynn/ab_smoke_ffncache

tinynn/ab_smoke_ffncache: tinynn/ab_smoke_ffncache.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_ffncache.rb -o tinynn/ab_smoke_ffncache

# ggml-native AdamW step (opt_step_adamw) parity vs project's plain-Adam.
ab-smoke-adamw-op: tinynn/ab_smoke_adamw_op
	./tinynn/ab_smoke_adamw_op

tinynn/ab_smoke_adamw_op: tinynn/ab_smoke_adamw_op.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adamw_op.rb -o tinynn/ab_smoke_adamw_op

# Persistent-tensor architecture check: data uploaded to a ctx_w tensor
# survives a compute cycle.
ab-smoke-persistent: tinynn/ab_smoke_persistent
	./tinynn/ab_smoke_persistent

tinynn/ab_smoke_persistent: tinynn/ab_smoke_persistent.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_persistent.rb -o tinynn/ab_smoke_persistent

# Dual-cgraph + persistent-weights design check: forward reads t_w;
# adam mutates t_w in place; forward sees the new value.
ab-smoke-dual-graph: tinynn/ab_smoke_dual_graph
	./tinynn/ab_smoke_dual_graph

tinynn/ab_smoke_dual_graph: tinynn/ab_smoke_dual_graph.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_dual_graph.rb -o tinynn/ab_smoke_dual_graph

# M2 foundation: view_2d + cpy to write a single row into a persistent
# (max_T, d_head) KV buffer at a runtime-baked position.
ab-smoke-kv-write: tinynn/ab_smoke_kv_write
	./tinynn/ab_smoke_kv_write

tinynn/ab_smoke_kv_write: tinynn/ab_smoke_kv_write.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_write.rb -o tinynn/ab_smoke_kv_write

# M2 prototype: single-step decode through a KV cache. Pre-fills K/V
# for positions 0..POS-1, writes k_new/v_new at POS, computes scores
# + soft_max_ext + head_out. Parity vs hand-rolled native.
ab-smoke-kv-attn: tinynn/ab_smoke_kv_attn
	./tinynn/ab_smoke_kv_attn

tinynn/ab_smoke_kv_attn: tinynn/ab_smoke_kv_attn.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_attn.rb -o tinynn/ab_smoke_kv_attn

# M1.2: full single-block forward through the persistent graph.
# Parity vs native TransformerLM.forward() at n_layers=1, n_heads=2.
ab-smoke-full-forward-block: tinynn/ab_smoke_full_forward_block
	./tinynn/ab_smoke_full_forward_block

tinynn/ab_smoke_full_forward_block: tinynn/ab_smoke_full_forward_block.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_full_forward_block.rb -o tinynn/ab_smoke_full_forward_block

# Wallclock bench: native TransformerLM.forward vs FullForwardFFICache.
full-forward-bench: tinynn/full_forward_bench
	./tinynn/full_forward_bench

tinynn/full_forward_bench: tinynn/full_forward_bench.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/full_forward_bench.rb -o tinynn/full_forward_bench

full-forward-bench-cuda: tinynn/full_forward_bench_cuda
	./tinynn/full_forward_bench_cuda

tinynn/full_forward_bench_cuda: tinynn/full_forward_bench_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/full_forward_bench_cuda.rb -o tinynn/full_forward_bench_cuda

ab-smoke-dual-graph-cuda: tinynn/ab_smoke_dual_graph_cuda
	./tinynn/ab_smoke_dual_graph_cuda

tinynn/ab_smoke_dual_graph_cuda: tinynn/ab_smoke_dual_graph_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_dual_graph_cuda.rb -o tinynn/ab_smoke_dual_graph_cuda

ab-smoke-adamw-op-cuda: tinynn/ab_smoke_adamw_op_cuda
	./tinynn/ab_smoke_adamw_op_cuda

tinynn/ab_smoke_adamw_op_cuda: tinynn/ab_smoke_adamw_op_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_adamw_op_cuda.rb -o tinynn/ab_smoke_adamw_op_cuda

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

# Walks every tensor in data/distilgpt2-f32.gguf via tnn_gguf_*. Used to
# confirm large HF-converted GGUFs roundtrip through the project FFI.
gguf-inspect: tinynn/gguf_inspect
	./tinynn/gguf_inspect

tinynn/gguf_inspect: tinynn/gguf_inspect.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_inspect.rb -o tinynn/gguf_inspect

# GPT2LM build smoke: confirm lib/gpt2.rb Spinel-compiles and the
# forward shapes line up. Toy dims, random weights — values mean nothing.
gpt2-build-smoke: tinynn/gpt2_build_smoke
	./tinynn/gpt2_build_smoke

tinynn/gpt2_build_smoke: tinynn/gpt2_build_smoke.rb lib/transformer.rb lib/gpt2.rb
	$(SPINEL) tinynn/gpt2_build_smoke.rb -o tinynn/gpt2_build_smoke

# Load distilgpt2-f32.gguf into a GPT2LM and print sentinel weights
# per category. Verifies name mapping + per-head split before forward.
gpt2-load-smoke: tinynn/gpt2_load_smoke
	./tinynn/gpt2_load_smoke

tinynn/gpt2_load_smoke: tinynn/gpt2_load_smoke.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_load_smoke.rb -o tinynn/gpt2_load_smoke

# End-to-end distilgpt2 inference demo. Reads pre-tokenized IDs from
# data/prompt_ids.txt, loads weights from data/distilgpt2-f32.gguf,
# greedy-generates N_NEW tokens via native Mat forward, writes the
# full ID sequence back. Decode with prep/tokens.py decode.
distilgpt2_demo: distilgpt2_demo.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# FFI persistent-graph variant of distilgpt2_demo. Same I/O contract.
distilgpt2_demo_ffi: distilgpt2_demo_ffi.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# KV-cache variant: per-step decode (constant in prompt length).
distilgpt2_demo_kv: distilgpt2_demo_kv.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi_kv.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Parity probe: one forward at distilgpt2 shape, dump last-row logits
# to data/ours_logits.txt. Pair with prep/parity.py for the HF reference.
gpt2-parity: tinynn/gpt2_parity
	./tinynn/gpt2_parity

tinynn/gpt2_parity: tinynn/gpt2_parity.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_parity.rb -o tinynn/gpt2_parity

# FFI parity probe: persistent ggml graph with LayerNorm + biases.
# Dumps last-row logits to data/ours_ffi_logits.txt.
gpt2-ffi-parity: tinynn/gpt2_ffi_parity
	./tinynn/gpt2_ffi_parity

tinynn/gpt2_ffi_parity: tinynn/gpt2_ffi_parity.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_ffi_parity.rb -o tinynn/gpt2_ffi_parity

# Apples-to-apples bench: native Mat vs FFI on the same forward.
# Re-encode data/prompt_ids.txt first so prompt length matches T_SEQ=5.
gpt2-bench: tinynn/gpt2_bench
	./tinynn/gpt2_bench

tinynn/gpt2_bench: tinynn/gpt2_bench.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi.rb lib/gpt2_ffi_kv.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_bench.rb -o tinynn/gpt2_bench

# KV-cache parity probe: prefill the prompt one token at a time through
# GPT2KVFFICache, dump last-position logits.
gpt2-kv-parity: tinynn/gpt2_kv_parity
	./tinynn/gpt2_kv_parity

tinynn/gpt2_kv_parity: tinynn/gpt2_kv_parity.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi_kv.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_kv_parity.rb -o tinynn/gpt2_kv_parity

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

tinynn/persistent_bench_cuda: tinynn/persistent_bench_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
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

tinynn/ab_smoke_cuda: tinynn/ab_smoke_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cuda.rb -o tinynn/ab_smoke_cuda

# Consolidated CUDA parity test: matmul + add + gelu + rms_norm + softmax + scale + ffn_pipeline.
ab-smoke-all-cuda: tinynn/ab_smoke_all_cuda
	./tinynn/ab_smoke_all_cuda

tinynn/ab_smoke_all_cuda: tinynn/ab_smoke_all_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_all_cuda.rb -o tinynn/ab_smoke_all_cuda

# Transformer-shape parity + wallclock bench on CUDA (GB10).
ab-smoke-big-cuda: tinynn/ab_smoke_big_cuda
	./tinynn/ab_smoke_big_cuda

tinynn/ab_smoke_big_cuda: tinynn/ab_smoke_big_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
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
