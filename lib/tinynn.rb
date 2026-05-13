# FFI bridge between the project's Mat (row-major f64) and ggml's
# backend-aware tensor library. Loads the static archives produced by
# the Makefile's `setup-ggml` (or `setup-ggml-cuda`) target.
#
# Usage (from a driver script):
#
#   require_relative "lib/transformer"   # defines Mat
#   require_relative "lib/tinynn"        # adds TinyNN.matmul(a, b)
#
# TinyNN.matmul(a, b) computes a · b and returns a fresh Mat with the
# same row-major layout. Internally it spins up a one-shot ggml session,
# uploads, computes, downloads, frees. Performance is bad for many small
# calls (kernel-launch + backend-init per call); the eventual fix is to
# share a persistent session across the training step. For S2 — a single
# A/B smoke check — one-shot is fine.

module TinyNN
  ffi_lib "tinynn_ggml"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  ffi_lib "gomp"
  # spinel wrapper adds -lm *before* FFI libs; ggml-cpu needs it again.
  ffi_lib "m"

  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build/src -Lvendor/ggml/build/src/ggml-cpu"

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

  # a · b where both are project Mats (row-major f64). Returns a Mat
  # (rows = a.nrows, cols = b.ncols).
  #
  # Implementation note: ggml_mul_mat computes A · B^T. To get A · B we
  # upload b TRANSPOSED — b is (br x bc) row-major; we present it to
  # ggml as a (bc x br) tensor whose rows are b's columns. Then ggml's
  # A · B^T = A · B (because the "B^T" inside ggml lines up with the
  # original b shape).
  def self.matmul(a, b)
    sess = TinyNN.tnn_session_new(0)   # 0 = CPU; flip to 1 for CUDA when built

    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    # ggml-side tensor for b^T: rows=b.ncols, cols=b.nrows.
    tb_t = TinyNN.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNN.tnn_matmul(sess, ta, tb_t)
    TinyNN.tnn_realize(sess, tc)

    # Upload a (row-major flat).
    i = 0
    na = a.nrows * a.ncols
    while i < na
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    # Upload b TRANSPOSED into scratch: scratch[j*b.nrows + i] = b[i,j].
    bc = b.ncols
    br = b.nrows
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNN.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tb_t)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result tensor ggml shape: ne0=m=a.nrows, ne1=n=b.ncols. Read into
    # row-major Mat[i][j] (= flat[i*ncols+j]) from scratch[j*m + i].
    out = Mat.new(a.nrows, b.ncols)
    m = a.nrows
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNN.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end
end
