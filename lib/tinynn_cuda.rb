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

# Same AdamStepResult definition as lib/tinynn.rb — drivers require
# exactly one of {tinynn, tinynn_cuda} so we duplicate to keep each
# file self-sufficient.
class AdamStepResult
  attr_accessor :param, :mom_m, :mom_v
  def initialize(param, mom_m, mom_v)
    @param = param
    @mom_m = mom_m
    @mom_v = mom_v
  end
end

# Same as FFNFFICache in lib/tinynn.rb but uses TinyNNCuda. The class
# name differs so that drivers requiring BOTH modules (e.g. the CUDA
# parity smoke loads tinynn_cuda directly, and lib/transformer.rb
# transitively pulls in tinynn) don't trip Spinel's same-class-defined-
# twice path. For CUDA training, sed-swap `FFNFFICache` to
# `FFNFFICacheCuda` in lib/transformer.rb along with `TinyNN.` to
# `TinyNNCuda.` in feed_forward_ffi.
class FFNFFICacheCuda
  attr_accessor :sess, :t_h, :t_w1_t, :t_w2_t,
                :t_pre, :t_hidden, :t_out,
                :t_seq, :d_model, :d_ff, :realized

  def initialize
    @realized = false
    @t_seq    = 0
    @d_model  = 0
    @d_ff     = 0
    @sess     = nil
    @t_h      = nil
    @t_w1_t   = nil
    @t_w2_t   = nil
    @t_pre    = nil
    @t_hidden = nil
    @t_out    = nil
  end

  def realize_for(t_seq, d_model, d_ff)
    @t_seq   = t_seq
    @d_model = d_model
    @d_ff    = d_ff

    @sess   = TinyNNCuda.tnn_session_new(1)
    @t_h    = TinyNNCuda.tnn_input_2d_f32(@sess, t_seq,  d_model)
    @t_w1_t = TinyNNCuda.tnn_input_2d_f32(@sess, d_ff,   d_model)
    @t_w2_t = TinyNNCuda.tnn_input_2d_f32(@sess, d_model, d_ff)

    @t_pre    = TinyNNCuda.tnn_matmul(@sess, @t_w1_t, @t_h)
    @t_hidden = TinyNNCuda.tnn_gelu(@sess, @t_pre)
    @t_out    = TinyNNCuda.tnn_matmul(@sess, @t_w2_t, @t_hidden)
    TinyNNCuda.tnn_set_output(@t_pre)
    TinyNNCuda.tnn_set_output(@t_hidden)
    TinyNNCuda.tnn_set_output(@t_out)
    TinyNNCuda.tnn_realize(@sess, @t_out)

    @realized = true
  end
end

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
  ffi_func :tnn_upload_from_float_array, [:ptr, :ptr, :float_array, :size_t], :int
  ffi_func :tnn_upload_from_int_array,   [:ptr, :ptr, :int_array,   :size_t], :int
  # CPU-scratch-backed custom kernels (gelu_back, adam_step). They
  # operate on host memory regardless of the session's backend; what
  # makes them "CUDA-mirrored" is just exposing them under TinyNNCuda
  # for callers using a CUDA session.
  ffi_func :tnn_gelu_back_scratch,[:ptr, :int],             :void
  ffi_func :tnn_set_output,       [:ptr],                   :void
  ffi_func :tnn_adam_step_scratch,[:ptr, :int, :double, :double, :double, :double, :double, :double], :void
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

  # GeLU backward (tanh approx) via custom CPU kernel. Mirrors
  # TinyNN.gelu_back; same scratch-layout protocol.
  def self.gelu_back(x, dh)
    sess = TinyNNCuda.tnn_session_new(1)
    n = x.nrows * x.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, n + i, dh.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_gelu_back_scratch(sess, n)
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 2 * n + i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    out
  end

  # cross_entropy_grad = (softmax(logits) - one_hot(targets)) / n_pred.
  # Composable from TinyNNCuda.softmax + scale + add.
  def self.cross_entropy_grad(logits, targets, n_pred)
    oh = Mat.new(logits.nrows, logits.ncols)
    i = 0
    while i < n_pred
      oh.flat[i * logits.ncols + targets[i]] = 1.0
      i = i + 1
    end
    sm = TinyNNCuda.softmax(logits)
    inv_n = 1.0 / n_pred.to_f
    sm_s = TinyNNCuda.scale(sm, inv_n)
    oh_s = TinyNNCuda.scale(oh, -inv_n)
    TinyNNCuda.add(sm_s, oh_s)
  end

  # Adam step via custom CPU kernel. Returns AdamStepResult (param,
  # mom_m, mom_v) — same shape as TinyNN.adam_step.
  def self.adam_step(param, grad, m, v, lr, b1, b2, eps, omc1, omc2)
    sess = TinyNNCuda.tnn_session_new(1)
    n = param.nrows * param.ncols
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, i, param.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, n + i, grad.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, 2 * n + i, m.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNNCuda.tnn_scratch_set(sess, 3 * n + i, v.flat[i])
      i = i + 1
    end
    TinyNNCuda.tnn_adam_step_scratch(sess, n, lr, b1, b2, eps, omc1, omc2)
    new_param = Mat.new(param.nrows, param.ncols)
    new_mom_m = Mat.new(param.nrows, param.ncols)
    new_mom_v = Mat.new(param.nrows, param.ncols)
    i = 0
    while i < n
      new_param.flat[i] = TinyNNCuda.tnn_scratch_get(sess, i)
      new_mom_m.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 2 * n + i)
      new_mom_v.flat[i] = TinyNNCuda.tnn_scratch_get(sess, 3 * n + i)
      i = i + 1
    end
    TinyNNCuda.tnn_session_free(sess)
    AdamStepResult.new(new_param, new_mom_m, new_mom_v)
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
    TinyNNCuda.tnn_upload_from_float_array(sess, tensor, mat.flat, mat.nrows * mat.ncols)
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

  # Alias to match the CPU module's name; used by feed_forward_ffi.
  def self.stage_transposed_and_upload(sess, target, b)
    TinyNNCuda.upload_transposed(sess, target, b)
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
