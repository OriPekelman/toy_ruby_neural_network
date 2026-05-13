# CUDA-flavored FFI bridge (sibling of lib/tinynn.rb).
#
# Same API surface as lib/tinynn.rb — same `TinyNNCuda.matmul(a, b)`
# semantics — but links against the CUDA-built ggml backend and asks
# tnn_session_new for the CUDA device. Drivers that need CPU use
# `require_relative "lib/tinynn"`; drivers that need GPU use
# `require_relative "lib/tinynn_cuda"`. We deliberately do *not* try to
# pick at runtime: spinel's FFI library linkage is static so a single
# binary either has the CUDA archives linked in or it doesn't.
#
# Build: `make ab-smoke-cuda` (requires `make setup-ggml-cuda` to have
# produced vendor/ggml/build-cuda first).

module TinyNNCuda
  ffi_lib "tinynn_ggml_cuda"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-cuda"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  ffi_lib "gomp"
  ffi_lib "m"
  # CUDA: static runtime + cublas, plus the dynamic driver lib.
  ffi_lib "cublas_static"
  ffi_lib "cublasLt_static"
  ffi_lib "culibos"
  ffi_lib "cudart_static"
  ffi_lib "cuda"
  ffi_lib "rt"
  ffi_lib "dl"

  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build-cuda/src -Lvendor/ggml/build-cuda/src/ggml-cpu -Lvendor/ggml/build-cuda/src/ggml-cuda -L/usr/local/cuda/lib64"

  ffi_func :tnn_session_new,      [:int],                   :ptr
  ffi_func :tnn_session_free,     [:ptr],                   :void
  ffi_func :tnn_backend_name,     [:ptr],                   :str
  ffi_func :tnn_input_2d_f32,     [:ptr, :int, :int],       :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int
  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int

  def self.matmul(a, b)
    sess = TinyNNCuda.tnn_session_new(1)   # 1 = prefer CUDA

    ta   = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb_t = TinyNNCuda.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc   = TinyNNCuda.tnn_matmul(sess, ta, tb_t)
    TinyNNCuda.tnn_realize(sess, tc)

    i = 0
    na = a.nrows * a.ncols
    while i < na
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)

    bc = b.ncols
    br = b.nrows
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNNCuda.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb_t)

    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)

    out = Mat.new(a.nrows, b.ncols)
    m = a.nrows
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNNCuda.tnn_session_free(sess)
    out
  end
end
