# FFI bridge between the project's Mat (row-major f64) and ggml's
# backend-aware tensor library. Loads the static archives produced by
# the Makefile's `setup-ggml` (or `setup-ggml-cuda`) target.
#
# Usage (from a driver script):
#
#   require_relative "lib/transformer"   # defines Mat
#   require_relative "lib/tinynn"        # adds TinyNN.matmul(a, b)
#
# TinyNN.matmul(a, b) computes a ** b and returns a fresh Mat with the
# same row-major layout. Internally it spins up a one-shot ggml session,
# uploads, computes, downloads, frees. Performance is bad for many small
# calls (kernel-launch + backend-init per call); the eventual fix is to
# share a persistent session across the training step. For S2 *** a single
# A/B smoke check *** one-shot is fine.

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
  ffi_func :tnn_add,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_gelu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_rms_norm,         [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_transpose,        [:ptr, :ptr],             :ptr
  ffi_func :tnn_scale,            [:ptr, :ptr, :double],    :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int
  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int

  # a ** b where both are project Mats (row-major f64). Returns a Mat
  # (rows = a.nrows, cols = b.ncols).
  #
  # Implementation note: ggml_mul_mat computes A ** B^T. To get A ** B we
  # upload b TRANSPOSED *** b is (br x bc) row-major; we present it to
  # ggml as a (bc x br) tensor whose rows are b's columns. Then ggml's
  # A ** B^T = A ** B (because the "B^T" inside ggml lines up with the
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

  # Element-wise a + b. Both Mats must have the same shape.
  def self.add(a, b)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNN.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNN.tnn_add(sess, ta, tb)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tb)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result is row-major same shape as a (ne0=cols, ne1=rows, flat
    # is row-major already since ggml_add preserves layout).
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise GeLU (tanh approximation, matches project's feed_forward).
  def self.gelu(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_gelu(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # RMSNorm(x) * gamma. x is (T, d_model), gamma is Array<Float> of
  # length d_model. eps defaults to 1e-5 (matches the project's
  # rms_norm helper).
  def self.rms_norm(x, gamma, eps)
    sess = TinyNN.tnn_session_new(0)
    tx = TinyNN.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    # gamma as a 1-row tensor: shape (1, d_model). ggml will broadcast
    # across x's leading dimension during the mul.
    tg = TinyNN.tnn_input_2d_f32(sess, 1, x.ncols)
    tc = TinyNN.tnn_rms_norm(sess, tx, tg, eps)
    TinyNN.tnn_realize(sess, tc)

    # Upload x.
    nx = x.nrows * x.ncols
    i = 0
    while i < nx
      TinyNN.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tx)

    # Upload gamma (length d_model).
    i = 0
    while i < x.ncols
      TinyNN.tnn_scratch_set(sess, i, gamma[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tg)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < nx
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Per-row softmax. Matches the project's softmax_rows! (out-of-place).
  def self.softmax(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_softmax(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Transpose. Returns a Mat with rows/cols swapped.
  def self.transpose(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_transpose(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result shape: (a.ncols, a.nrows) *** rows and cols swapped.
    # ggml stores it contiguous after ggml_cont; row-major readout is
    # straightforward since the transposed tensor's ne0/ne1 already
    # match the target Mat's cols/rows.
    out = Mat.new(a.ncols, a.nrows)
    rin  = a.nrows
    cin  = a.ncols
    i = 0
    while i < cin
      j = 0
      while j < rin
        out.flat[i * rin + j] = TinyNN.tnn_scratch_get(sess, i * rin + j)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Internal: stage b TRANSPOSED into scratch, then bulk-upload to `target`.
  def self.stage_transposed_and_upload(sess, target, b)
    br = b.nrows
    bc = b.ncols
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNN.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNN.tnn_upload(sess, target)
  end

  # Internal: stage `m` row-major into scratch, then bulk-upload to `target`.
  def self.stage_row_major_and_upload(sess, target, m)
    n = m.nrows * m.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, m.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, target)
  end

  # FFN-shaped chain: result = gelu(h * w1) * w2.
  #
  # Calls three op-sized sessions, each reusing the cached engine (the
  # backend + scheduler init runs once, not three times). One ggml-graph
  # chaining is theoretically possible but needs explicit intermediate
  # transposes because mul_mat's result has ne0 swapped relative to the
  # next op's k-dim. Sticking to three sessions until we have a clean
  # chain-friendly layout convention.
  def self.ffn_pipeline(h, w1, w2)
    pre    = TinyNN.matmul(h, w1)
    hidden = TinyNN.gelu(pre)
    TinyNN.matmul(hidden, w2)
  end

  # a * b^T natively (matches Mat#matmul_t). Faster than .matmul(b) for the
  # same shapes because there's no Ruby-side transpose of b on upload.
  def self.matmul_t(a, b)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNN.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNN.tnn_matmul(sess, ta, tb)
    TinyNN.tnn_realize(sess, tc)

    TinyNN.stage_row_major_and_upload(sess, ta, a)
    TinyNN.stage_row_major_and_upload(sess, tb, b)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, b.nrows)
    m = a.nrows
    n = b.nrows
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

  # a^T * b (matches Mat#t_matmul). Both inputs uploaded transposed so
  # ggml's ne0 lines up with the summed-over K dimension.
  def self.t_matmul(a, b)
    sess = TinyNN.tnn_session_new(0)
    # Both tensors created as their transposed shape:
    #   ta_t: ne0=a.nrows (=K), ne1=a.ncols (=M)
    #   tb_t: ne0=b.nrows (=K), ne1=b.ncols (=N)
    ta_t = TinyNN.tnn_input_2d_f32(sess, a.ncols, a.nrows)
    tb_t = TinyNN.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNN.tnn_matmul(sess, ta_t, tb_t)
    TinyNN.tnn_realize(sess, tc)

    TinyNN.stage_transposed_and_upload(sess, ta_t, a)
    TinyNN.stage_transposed_and_upload(sess, tb_t, b)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.ncols, b.ncols)
    m = a.ncols
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

  # Element-wise a * s for scalar s. Returns a new Mat (out-of-place).
  def self.scale(a, s)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_scale(sess, ta, s)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end
end
