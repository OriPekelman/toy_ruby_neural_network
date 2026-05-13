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
  ffi_func :tnn_input_1d_i32,     [:ptr, :int],             :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_add,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_gelu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_rms_norm,         [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_scale,            [:ptr, :ptr, :double],    :ptr
  ffi_func :tnn_softmax_back,     [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_get_rows,         [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_get_rows_back,    [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_scratch_set_i32,  [:ptr, :int, :int],       :void
  ffi_func :tnn_scratch_get_i32,  [:ptr, :int],             :int
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

  def self.add(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNNCuda.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNNCuda.tnn_add(sess, ta, tb)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.gelu(a)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_gelu(sess, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.rms_norm(x, gamma, eps)
    sess = TinyNNCuda.tnn_session_new(1)
    tx = TinyNNCuda.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    tg = TinyNNCuda.tnn_input_2d_f32(sess, 1, x.ncols)
    tc = TinyNNCuda.tnn_rms_norm(sess, tx, tg, eps)
    TinyNNCuda.tnn_realize(sess, tc)
    nx = x.nrows * x.ncols
    i = 0
    while i < nx
      TinyNNCuda.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tx)
    i = 0
    while i < x.ncols
      TinyNNCuda.tnn_scratch_set(sess, i, gamma[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tg)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < nx
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.softmax(a)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_softmax(sess, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  def self.scale(a, s)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNNCuda.tnn_scale(sess, ta, s)
    TinyNNCuda.tnn_realize(sess, tc)
    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # gelu(h * w1) * w2 chained via the persistent CUDA engine.
  def self.ffn_pipeline(h, w1, w2)
    pre    = TinyNNCuda.matmul(h, w1)
    hidden = TinyNNCuda.gelu(pre)
    TinyNNCuda.matmul(hidden, w2)
  end

  # a * b^T (matches Mat#matmul_t).
  def self.matmul_t(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta = TinyNNCuda.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNNCuda.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNNCuda.tnn_matmul(sess, ta, tb)
    TinyNNCuda.tnn_realize(sess, tc)
    na = a.nrows * a.ncols
    i = 0
    while i < na
      TinyNNCuda.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    nb = b.nrows * b.ncols
    i = 0
    while i < nb
      TinyNNCuda.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tb)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a.nrows, b.nrows)
    m = a.nrows
    n = b.nrows
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

  # a^T * b (matches Mat#t_matmul). Both uploaded transposed.
  def self.t_matmul(a, b)
    sess = TinyNNCuda.tnn_session_new(1)
    ta_t = TinyNNCuda.tnn_input_2d_f32(sess, a.ncols, a.nrows)
    tb_t = TinyNNCuda.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNNCuda.tnn_matmul(sess, ta_t, tb_t)
    TinyNNCuda.tnn_realize(sess, tc)
    ar = a.nrows
    ac = a.ncols
    i = 0
    while i < ar
      j = 0
      while j < ac
        TinyNNCuda.tnn_scratch_set(sess, j * ar + i, a.flat[i * ac + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta_t)
    br = b.nrows
    bc = b.ncols
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
    out = Mat.new(a.ncols, b.ncols)
    m = a.ncols
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

  # Per-row softmax backward.
  def self.softmax_back(a_softmax, dy)
    sess = TinyNNCuda.tnn_session_new(1)
    tdy = TinyNNCuda.tnn_input_2d_f32(sess, dy.nrows, dy.ncols)
    ta  = TinyNNCuda.tnn_input_2d_f32(sess, a_softmax.nrows, a_softmax.ncols)
    tc  = TinyNNCuda.tnn_softmax_back(sess, tdy, ta)
    TinyNNCuda.tnn_realize(sess, tc)
    n = dy.nrows * dy.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, dy.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tdy)
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, a_softmax.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ta)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tc)
    out = Mat.new(a_softmax.nrows, a_softmax.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # Embedding lookup: gather rows.
  def self.embed_lookup(table, indices)
    n_idx = indices.length
    sess  = TinyNNCuda.tnn_session_new(1)
    ttab  = TinyNNCuda.tnn_input_2d_f32(sess, table.nrows, table.ncols)
    tidx  = TinyNNCuda.tnn_input_1d_i32(sess, n_idx)
    tout  = TinyNNCuda.tnn_get_rows(sess, ttab, tidx)
    TinyNNCuda.tnn_realize(sess, tout)
    nt = table.nrows * table.ncols
    i = 0
    while i < nt
      TinyNNCuda.tnn_scratch_set(sess, i, table.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, ttab)
    i = 0
    while i < n_idx
      TinyNNCuda.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tidx)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tout)
    out = Mat.new(n_idx, table.ncols)
    n = n_idx * table.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # Embedding scatter-add (backward).
  def self.embed_back(d_out, indices, vocab_size)
    n_idx = indices.length
    sess  = TinyNNCuda.tnn_session_new(1)
    td    = TinyNNCuda.tnn_input_2d_f32(sess, d_out.nrows, d_out.ncols)
    tidx  = TinyNNCuda.tnn_input_1d_i32(sess, n_idx)
    tshape = TinyNNCuda.tnn_input_2d_f32(sess, vocab_size, d_out.ncols)
    tout  = TinyNNCuda.tnn_get_rows_back(sess, td, tidx, tshape)
    TinyNNCuda.tnn_realize(sess, tout)
    nd = d_out.nrows * d_out.ncols
    i = 0
    while i < nd
      TinyNNCuda.tnn_scratch_set(sess, i, d_out.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, td)
    i = 0
    while i < n_idx
      TinyNNCuda.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tidx)
    TinyNNCuda.tnn_compute(sess)
    TinyNNCuda.tnn_download(sess, tout)
    out = Mat.new(vocab_size, d_out.ncols)
    n = vocab_size * d_out.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # SGD: param_new = param - lr * grad. Composed.
  def self.sgd_step(param, grad, lr)
    TinyNNCuda.add(param, TinyNNCuda.scale(grad, -lr))
  end

  # ----- Persistent-session API (mirrors TinyNN's; see lib/tinynn.rb) -----
  def self.persistent_new(prefer_cuda);  TinyNNCuda.tnn_session_new(prefer_cuda); end
  def self.persistent_free(sess);        TinyNNCuda.tnn_session_free(sess); end
  def self.alloc_2d(sess, r, c);         TinyNNCuda.tnn_input_2d_f32(sess, r, c); end
  def self.alloc_1d_i32(sess, n);        TinyNNCuda.tnn_input_1d_i32(sess, n); end
  def self.build_matmul(sess, ta, tb);   TinyNNCuda.tnn_matmul(sess, ta, tb); end
  def self.build_add(sess, ta, tb);      TinyNNCuda.tnn_add(sess, ta, tb); end
  def self.build_gelu(sess, ta);         TinyNNCuda.tnn_gelu(sess, ta); end
  def self.build_softmax(sess, ta);      TinyNNCuda.tnn_softmax(sess, ta); end
  def self.build_scale(sess, ta, s);     TinyNNCuda.tnn_scale(sess, ta, s); end
  def self.build_rms_norm(sess, x, g, e); TinyNNCuda.tnn_rms_norm(sess, x, g, e); end
  def self.realize(sess, r);             TinyNNCuda.tnn_realize(sess, r); end
  def self.compute(sess);                TinyNNCuda.tnn_compute(sess); end

  def self.upload_row_major(sess, tensor, mat)
    n = mat.nrows * mat.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, mat.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tensor)
  end

  def self.upload_transposed(sess, tensor, mat)
    br = mat.nrows
    bc = mat.ncols
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNNCuda.tnn_scratch_set(sess, j * br + i, mat.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNNCuda.tnn_upload(sess, tensor)
  end

  def self.download_row_major(sess, tensor, rows, cols)
    TinyNNCuda.tnn_download(sess, tensor)
    out = Mat.new(rows, cols)
    n = rows * cols
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      i = i + 1
    end
    out
  end

  def self.download_matmul(sess, tensor, m, n)
    TinyNNCuda.tnn_download(sess, tensor)
    out = Mat.new(m, n)
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNNCuda.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    out
  end
end
