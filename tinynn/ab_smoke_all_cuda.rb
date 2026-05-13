# Consolidated CUDA parity smoke: matmul, add, gelu, rms_norm, softmax,
# scale, ffn_pipeline. Compares each TinyNNCuda op to the native Mat
# implementation (or an inline reference for gelu/rms_norm/softmax).
# Run via `make ab-smoke-all-cuda` once `make setup-ggml-cuda` has
# produced vendor/ggml/build-cuda.

require_relative "../lib/transformer"
require_relative "../lib/tinynn_cuda"

# --- reference helpers ---

def gelu_native(x)
  c = 0.7978845608028654
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    v = x.flat[i]
    u = c * (v + 0.044715 * v * v * v)
    out.flat[i] = 0.5 * v * (1.0 + Math.tanh(u))
    i = i + 1
  end
  out
end

def softmax_native(x)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    m = x.flat[r * x.ncols]
    c = 1
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      if v > m
        m = v
      end
      c = c + 1
    end
    s = 0.0
    c = 0
    while c < x.ncols
      e = Math.exp(x.flat[r * x.ncols + c] - m)
      out.flat[r * x.ncols + c] = e
      s = s + e
      c = c + 1
    end
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = out.flat[r * x.ncols + c] / s
      c = c + 1
    end
    r = r + 1
  end
  out
end

def rms_norm_native(x, gamma, eps)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    sum_sq = 0.0
    c = 0
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      sum_sq = sum_sq + v * v
      c = c + 1
    end
    mean_sq = sum_sq / x.ncols.to_f
    rms = Math.sqrt(mean_sq + eps)
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = x.flat[r * x.ncols + c] / rms * gamma[c]
      c = c + 1
    end
    r = r + 1
  end
  out
end

def scale_native(a, s)
  out = Mat.new(a.nrows, a.ncols)
  n = a.nrows * a.ncols
  i = 0
  while i < n
    out.flat[i] = a.flat[i] * s
    i = i + 1
  end
  out
end

def cmp(name, native, ffi, tol)
  n = native.nrows * native.ncols
  ok = true
  i = 0
  while i < n
    d = native.flat[i] - ffi.flat[i]
    if d < 0
      d = -d
    end
    if d > tol
      ok = false
    end
    i = i + 1
  end
  puts name + ": match=" + ok.to_s + "  (n=" + n.to_s + ", tol=" + tol.to_s + ")"
end

# --- matmul: 2x3 @ 3x2 ---
a = Mat.new(2, 3)
a.flat[0] = 1.0; a.flat[1] = 2.0; a.flat[2] = 3.0
a.flat[3] = 4.0; a.flat[4] = 5.0; a.flat[5] = 6.0
b = Mat.new(3, 2)
b.flat[0] = 7.0;  b.flat[1] = 8.0
b.flat[2] = 9.0;  b.flat[3] = 10.0
b.flat[4] = 11.0; b.flat[5] = 12.0
cmp("matmul",  a.matmul(b), TinyNNCuda.matmul(a, b), 1.0e-4)

# --- add: 2x3 ---
b_add = Mat.new(2, 3)
b_add.flat[0] = -1.0; b_add.flat[1] = 0.5; b_add.flat[2] = 2.0
b_add.flat[3] =  3.0; b_add.flat[4] = -4.0; b_add.flat[5] = 5.0
cmp("add",     a.add(b_add),    TinyNNCuda.add(a, b_add), 1.0e-4)

# --- gelu: 1x5 ---
g = Mat.new(1, 5)
g.flat[0] = 0.0; g.flat[1] = 0.5; g.flat[2] = 3.0; g.flat[3] = -0.5; g.flat[4] = -3.0
cmp("gelu",    gelu_native(g),  TinyNNCuda.gelu(g),       1.0e-3)

# --- rms_norm: 2x4 ---
x = Mat.new(2, 4)
x.flat[0] = 1.0;  x.flat[1] = 2.0;  x.flat[2] = -1.0; x.flat[3] = 0.5
x.flat[4] = -0.5; x.flat[5] = 3.0;  x.flat[6] = 0.0;  x.flat[7] = 1.5
gamma = [1.0, 0.5, 2.0, 1.5]
eps = 0.00001
cmp("rms_norm",rms_norm_native(x, gamma, eps), TinyNNCuda.rms_norm(x, gamma, eps), 1.0e-3)

# --- softmax: 2x4 ---
sm = Mat.new(2, 4)
sm.flat[0] = 1.0; sm.flat[1] = 2.0; sm.flat[2] = 3.0; sm.flat[3] = 4.0
sm.flat[4] = 0.5; sm.flat[5] = 0.5; sm.flat[6] = 0.5; sm.flat[7] = 0.5
cmp("softmax", softmax_native(sm), TinyNNCuda.softmax(sm), 1.0e-4)

# --- scale: 2x3 ---
cmp("scale",   scale_native(a, 0.5), TinyNNCuda.scale(a, 0.5), 1.0e-5)

# --- ffn_pipeline: (3,4)*(4,6)*(6,4) ---
h_p  = Mat.new(3, 4)
w1_p = Mat.new(4, 6)
w2_p = Mat.new(6, 4)
n_h = 12
i = 0
while i < n_h
  h_p.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end
n_w1 = 24
i = 0
while i < n_w1
  w1_p.flat[i] = i.to_f * 0.05 - 0.3
  i = i + 1
end
n_w2 = 24
i = 0
while i < n_w2
  w2_p.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end
pre = h_p.matmul(w1_p)
hid = gelu_native(pre)
nat = hid.matmul(w2_p)
cmp("ffn_pipeline", nat, TinyNNCuda.ffn_pipeline(h_p, w1_p, w2_p), 1.0e-3)
