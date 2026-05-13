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

tinynn/libtinynn_ggml.a: tinynn/tinynn_ggml.o
	ar $(ARFLAGS) $@ $<

# --- smoke test -------------------------------------------------------------
# Builds tinynn/smoke.rb against the CPU shim. Requires `setup-ggml` to have
# been run once first.
smoke: tinynn/smoke
	./tinynn/smoke

tinynn/smoke: tinynn/smoke.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/smoke.rb -o tinynn/smoke

# A/B parity test: native Mat#matmul vs TinyNN.matmul (FFI, CPU).
ab-smoke: tinynn/ab_smoke
	./tinynn/ab_smoke

tinynn/ab_smoke: tinynn/ab_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke.rb -o tinynn/ab_smoke

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

# --- maintenance ------------------------------------------------------------
clean:
	rm -f train_minimal train_tinystories \
	      tinynn/tinynn_ggml.o tinynn/libtinynn_ggml.a \
	      tinynn/tinynn_ggml_cuda.o tinynn/libtinynn_ggml_cuda.a \
	      tinynn/smoke tinynn/ab_smoke tinynn/ab_smoke_cuda

distclean: clean
	rm -rf $(GGML_DIR)/build $(GGML_DIR)/build-cuda

.PHONY: all clean distclean setup-ggml setup-ggml-cuda smoke
