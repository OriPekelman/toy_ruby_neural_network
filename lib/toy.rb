# lib/toy.rb — pleasant transformer building blocks.
#
# A thin sugar layer on top of Mat (lib/transformer.rb). Each class
# here owns its own parameters as ivars and has one entry point —
# `call(x)`. Models compose these so the forward pass reads like the
# transformer paper: pre-norm, attention with residual, FFN with
# residual.
#
# Shapes throughout follow this convention in comments:
#   [T, D]   T = sequence length     D = d_model
#   [T, Dh]  Dh = d_head             H = n_heads
#   [V]      V = vocab               Df = d_ff
#
# Spinel notes (relevant if you read the source):
#   • Each class's ivars are concrete types (Mat or Array<Float>).
#     No polymorphic containers.
#   • Per-head Q/K/V uses the literal-seed + push pattern
#     (Array<Mat>) — same as lib/transformer.rb's Block.
#   • Methods return concrete types (always Mat for tensor-valued ops)
#     so Spinel's type inference flows cleanly.

require_relative "transformer"   # Mat lives here

module Toy
  # =========================================================================
  # Toy::LayerNorm
  #   y = (x - mean) / sqrt(var + eps) * gamma + beta,  row-wise.
  #   gamma, beta: length D. eps default 1e-5 (HF GPT-2).
  # =========================================================================
  class LayerNorm
    attr_accessor :gamma, :beta, :d, :eps

    def initialize(d)
      @d     = d
      @eps   = 1.0e-5
      @gamma = Array.new(d, 1.0)
      @beta  = Array.new(d, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      t   = x.nrows
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        # mean over row
        sum = 0.0
        j = 0
        while j < d
          sum = sum + x.flat[i * d + j]
          j += 1
        end
        mean = sum / d
        # variance over row
        sumsq = 0.0
        j = 0
        while j < d
          v = x.flat[i * d + j] - mean
          sumsq = sumsq + v * v
          j += 1
        end
        inv = 1.0 / Math.sqrt(sumsq / d + @eps)
        # normalize, scale, shift
        j = 0
        while j < d
          n = (x.flat[i * d + j] - mean) * inv
          out.flat[i * d + j] = n * @gamma[j] + @beta[j]
          j += 1
        end
        i += 1
      end
      out
    end
  end

  # =========================================================================
  # Toy::Linear
  #   y = x · W + b  (b optional; bias=false skips it).
  # =========================================================================
  class Linear
    attr_accessor :w, :b, :has_bias, :in_dim, :out_dim

    def initialize(in_dim, out_dim, with_bias)
      @in_dim   = in_dim
      @out_dim  = out_dim
      @has_bias = with_bias
      @w        = Mat.new(in_dim, out_dim)
      @b        = Array.new(out_dim, 0.0)
    end

    # x: [T, in_dim] → [T, out_dim]
    def forward(x)
      out = x.matmul(@w)
      if @has_bias
        Toy.add_bias!(out, @b)
      end
      out
    end
  end

  # =========================================================================
  # Toy::Embedding
  #   Lookup table: row[i] = weight[i, :].
  # =========================================================================
  class Embedding
    attr_accessor :weight, :vocab, :d

    def initialize(vocab, d)
      @vocab  = vocab
      @d      = d
      @weight = Mat.new(vocab, d)
    end

    # ids: [T] (Array<Int>) → [T, D]
    def lookup(ids)
      t   = ids.length
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        row = ids[i]
        j = 0
        while j < d
          out.flat[i * d + j] = @weight.flat[row * d + j]
          j += 1
        end
        i += 1
      end
      out
    end

    # Slice rows [start, start + count) as a fresh Mat.
    # Used for absolute position embeddings.
    def slice(start, count)
      d   = @d
      out = Mat.new(count, d)
      i = 0
      while i < count
        j = 0
        while j < d
          out.flat[i * d + j] = @weight.flat[(start + i) * d + j]
          j += 1
        end
        i += 1
      end
      out
    end
  end

  # =========================================================================
  # Toy::CausalSelfAttention
  #
  #   Per-head storage (Array<Mat> for Q/K/V) matches lib/transformer.rb's
  #   Block layout, so the GGUF loader's split-heads helper works.
  #
  #     attn(x) =
  #       q_h = (x · W_q^h + b_q^h)       for h in 0..H
  #       k_h = (x · W_k^h + b_k^h)
  #       v_h = (x · W_v^h + b_v^h)
  #       s_h = softmax(causal(q_h · k_h^T / sqrt(Dh))) · v_h
  #       y   = hstack(s_0..s_H) · W_o + b_o
  # =========================================================================
  class CausalSelfAttention
    attr_accessor :w_q, :w_k, :w_v, :b_q, :b_k, :b_v, :w_o, :b_o,
                  :n_heads, :d_model, :d_head, :inv_sqrt

    def initialize(d_model, n_heads)
      @d_model  = d_model
      @n_heads  = n_heads
      @d_head   = d_model / n_heads
      @inv_sqrt = 1.0 / Math.sqrt(@d_head)

      # Per-head: literal-seed + push so Spinel types as PtrArray of Mat /
      # PtrArray of FloatArray.
      @w_q = [Mat.new(d_model, @d_head)]
      @w_k = [Mat.new(d_model, @d_head)]
      @w_v = [Mat.new(d_model, @d_head)]
      @b_q = [Array.new(@d_head, 0.0)]
      @b_k = [Array.new(@d_head, 0.0)]
      @b_v = [Array.new(@d_head, 0.0)]
      h = 1
      while h < n_heads
        @w_q.push(Mat.new(d_model, @d_head))
        @w_k.push(Mat.new(d_model, @d_head))
        @w_v.push(Mat.new(d_model, @d_head))
        @b_q.push(Array.new(@d_head, 0.0))
        @b_k.push(Array.new(@d_head, 0.0))
        @b_v.push(Array.new(@d_head, 0.0))
        h += 1
      end
      @w_o = Mat.new(d_model, d_model)
      @b_o = Array.new(d_model, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      head0    = head(x, 0)               # [T, Dh]
      per_head = [head0]
      h = 1
      while h < @n_heads
        per_head.push(head(x, h))         # [T, Dh] each
        h += 1
      end
      concat = Toy.hstack_heads(per_head, @n_heads, @d_head, @d_model)  # [T, D]
      out    = concat.matmul(@w_o)        # [T, D]
      Toy.add_bias!(out, @b_o)
      out
    end

    # One attention head. x: [T, D] → [T, Dh].
    def head(x, h)
      q = x.matmul(@w_q[h])
      Toy.add_bias!(q, @b_q[h])           # [T, Dh]
      k = x.matmul(@w_k[h])
      Toy.add_bias!(k, @b_k[h])           # [T, Dh]
      v = x.matmul(@w_v[h])
      Toy.add_bias!(v, @b_v[h])           # [T, Dh]

      scores = q.matmul_t(k)              # [T, T]
      scores.scale!(@inv_sqrt)
      Toy.causal_mask!(scores)
      Toy.softmax_rows!(scores)
      scores.matmul(v)                    # [T, Dh]
    end
  end

  # =========================================================================
  # Toy::FFN
  #   GPT-2 MLP: y = up · W2 + b2 where up = gelu(x · W1 + b1).
  #   `act` selects the activation: :gelu_new (tanh approx, HF default)
  #   or :gelu_exact (erf-based).
  # =========================================================================
  class FFN
    attr_accessor :w1, :w2, :b1, :b2, :d_model, :d_ff, :act

    def initialize(d_model, d_ff, act_sym)
      @d_model = d_model
      @d_ff    = d_ff
      @act     = act_sym                  # :gelu_new
      @w1 = Mat.new(d_model, d_ff)
      @w2 = Mat.new(d_ff,    d_model)
      @b1 = Array.new(d_ff,   0.0)
      @b2 = Array.new(d_model, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      pre = x.matmul(@w1)                 # [T, Df]
      Toy.add_bias!(pre, @b1)
      hidden = Toy.gelu_new(pre)          # [T, Df]
      out = hidden.matmul(@w2)            # [T, D]
      Toy.add_bias!(out, @b2)
      out
    end
  end

  # =========================================================================
  # Toy::RMSNorm — root-mean-square LayerNorm. No mean subtraction, no
  # beta. Standard in llama-family models.
  #
  #   y = x / sqrt(mean(x^2) + eps) * gamma   (row-wise)
  # =========================================================================
  class RMSNorm
    attr_accessor :gamma, :d, :eps

    # eps defaults to 1.0e-5 (matches Llama / SmolLM2). Override via
    # `rms.eps = ...` after construction — seeding the ivar with a Float
    # literal pins Spinel's type inference.
    def initialize(d)
      @d     = d
      @eps   = 1.0e-5
      @gamma = Array.new(d, 1.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      t   = x.nrows
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        sumsq = 0.0
        j = 0
        while j < d
          v = x.flat[i * d + j]
          sumsq = sumsq + v * v
          j += 1
        end
        inv = 1.0 / Math.sqrt(sumsq / d + @eps)
        j = 0
        while j < d
          out.flat[i * d + j] = x.flat[i * d + j] * inv * @gamma[j]
          j += 1
        end
        i += 1
      end
      out
    end
  end

  # =========================================================================
  # Toy::SwiGLU — gated FFN used by Llama / SmolLM2 / Qwen2 / Phi.
  #
  #   gate(x) = x · W_gate       up(x) = x · W_up
  #   y       = (silu(gate(x)) * up(x)) · W_down
  #
  # Three linear layers, no bias by default (llama convention). Element-
  # wise multiply between silu(gate) and up before the down projection.
  # =========================================================================
  class SwiGLU
    attr_accessor :w_gate, :w_up, :w_down, :d_model, :d_ff

    def initialize(d_model, d_ff)
      @d_model = d_model
      @d_ff    = d_ff
      @w_gate  = Mat.new(d_model, d_ff)
      @w_up    = Mat.new(d_model, d_ff)
      @w_down  = Mat.new(d_ff,    d_model)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      gate = x.matmul(@w_gate)              # [T, Df]
      up   = x.matmul(@w_up)                # [T, Df]
      Toy.silu!(gate)                       # [T, Df]
      Toy.hadamard!(gate, up)               # [T, Df]  (gate := gate * up)
      gate.matmul(@w_down)                  # [T, D]
    end
  end

  # =========================================================================
  # Free-standing helpers. These operate on Mat in place where possible
  # so they read like verbs: `causal_mask!`, `add_bias!`, `softmax_rows!`.
  # =========================================================================

  # SiLU activation, in-place. silu(x) = x / (1 + exp(-x)).
  def self.silu!(m)
    n = m.nrows * m.ncols
    i = 0
    while i < n
      v = m.flat[i]
      m.flat[i] = v / (1.0 + Math.exp(-v))
      i += 1
    end
  end

  # Elementwise multiply, into `dst` (dst := dst * src). Both have
  # identical shape. Param names avoid `a` / `b` to dodge a Spinel
  # collapse with TinyNN.matmul(a, b) and friends.
  def self.hadamard!(dst, src)
    n = dst.nrows * dst.ncols
    i = 0
    while i < n
      dst.flat[i] = dst.flat[i] * src.flat[i]
      i += 1
    end
  end

  # x[i, j] += b[j], in-place.
  def self.add_bias!(x, b)
    t = x.nrows
    d = x.ncols
    i = 0
    while i < t
      j = 0
      while j < d
        x.flat[i * d + j] = x.flat[i * d + j] + b[j]
        j += 1
      end
      i += 1
    end
  end

  # scores[i, j] = -inf  for j > i. In-place.
  def self.causal_mask!(scores)
    t = scores.nrows
    n = scores.ncols
    i = 0
    while i < t
      j = i + 1
      while j < n
        scores.flat[i * n + j] = -1.0e30
        j += 1
      end
      i += 1
    end
  end

  # Row-wise softmax, in-place, numerically stable (max-shift).
  def self.softmax_rows!(m)
    t = m.nrows
    n = m.ncols
    i = 0
    while i < t
      base = i * n
      mx = m.flat[base]
      j = 1
      while j < n
        v = m.flat[base + j]
        if v > mx
          mx = v
        end
        j += 1
      end
      sum = 0.0
      j = 0
      while j < n
        e = Math.exp(m.flat[base + j] - mx)
        m.flat[base + j] = e
        sum = sum + e
        j += 1
      end
      j = 0
      while j < n
        m.flat[base + j] = m.flat[base + j] / sum
        j += 1
      end
      i += 1
    end
  end

  # GeLU (tanh approximation — HF's gelu_new). Returns a fresh Mat.
  #   y = 0.5 x (1 + tanh( sqrt(2/π) (x + 0.044715 x^3) ))
  def self.gelu_new(x)
    n   = x.nrows * x.ncols
    out = Mat.new(x.nrows, x.ncols)
    c = 0.7978845608028654       # sqrt(2/π)
    i = 0
    while i < n
      v = x.flat[i]
      u = c * (v + 0.044715 * v * v * v)
      out.flat[i] = 0.5 * v * (1.0 + Math.tanh(u))
      i += 1
    end
    out
  end

  # Pack n_heads × [T, Dh] back into a single [T, D] matrix where
  # head h occupies columns [h*Dh, (h+1)*Dh).
  def self.hstack_heads(per_head, n_heads, d_head, d_model)
    t   = per_head[0].nrows
    out = Mat.new(t, d_model)
    h = 0
    while h < n_heads
      head = per_head[h]
      base = h * d_head
      i = 0
      while i < t
        j = 0
        while j < d_head
          out.flat[i * d_model + (base + j)] = head.flat[i * d_head + j]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end
end
