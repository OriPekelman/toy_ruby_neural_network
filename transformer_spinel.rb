# Spinel-compatible variant of transformer.rb.
#
#   • Mat is a class with flat 1D Float storage (Spinel can't currently
#     infer "Array of FloatArray" from `Array.new(n) { Array.new(m, 0.0) }`).
#   • Block / cache / KV cache structures are classes, not hashes.
#   • Arrays of objects are built with `[seed] + push` (Spinel's
#     `Array.new(n) { ... }` takes an IntArray-only fast path).
#   • Field names: `nrows`/`ncols` instead of `rows`/`cols` because the
#     latter triggers a name-collision in Spinel's type inference.
#
# This file is a smoke test of the patterns. It builds a tiny model state
# and runs one matmul to verify Spinel can compile it. The full forward +
# backward pass is in transformer.rb (CRuby-only).

# ----------------------------------------------------------------------------
#   Mat: a 2D matrix backed by a flat Float array, indexed at i*ncols+j.
# ----------------------------------------------------------------------------
class Mat
  attr_accessor :nrows, :ncols, :flat

  def initialize(nrows, ncols)
    @nrows = nrows
    @ncols = ncols
    @flat  = Array.new(nrows * ncols, 0.0)
  end

  def fill_random(scale)
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = (rand * 2 - 1) * scale
      i += 1
    end
  end

  # (m × n) · (n × p) → (m × p)
  def matmul(other)
    m = @nrows
    n = @ncols
    p = other.ncols
    out = Mat.new(m, p)
    i = 0
    while i < m
      k = 0
      while k < n
        aik = @flat[i * n + k]
        if aik != 0.0
          j = 0
          while j < p
            out.flat[i * p + j] += aik * other.flat[k * p + j]
            j += 1
          end
        end
        k += 1
      end
      i += 1
    end
    out
  end

  def transpose
    out = Mat.new(@ncols, @nrows)
    i = 0
    while i < @nrows
      j = 0
      while j < @ncols
        out.flat[j * @nrows + i] = @flat[i * @ncols + j]
        j += 1
      end
      i += 1
    end
    out
  end

  def add(other)
    out = Mat.new(@nrows, @ncols)
    n = @nrows * @ncols
    i = 0
    while i < n
      out.flat[i] = @flat[i] + other.flat[i]
      i += 1
    end
    out
  end

  def add!(other)
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] += other.flat[i]
      i += 1
    end
    self
  end

  def scale!(s)
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] *= s
      i += 1
    end
    self
  end
end

# ----------------------------------------------------------------------------
#   Block: a transformer block's parameters as a class (not a hash).
# ----------------------------------------------------------------------------
class Block
  attr_accessor :norm1_gamma, :w_q, :w_k, :w_v, :w_o,
                :norm2_gamma, :w_ff1, :w_ff2

  def initialize(d_model, d_head, d_ff, n_heads)
    s = 1.0 / Math.sqrt(d_model)

    @norm1_gamma = Array.new(d_model, 1.0)
    @norm2_gamma = Array.new(d_model, 1.0)

    # Per-head Q/K/V matrices: arrays of Mat (use seed-then-push).
    @w_q = build_head_mats(n_heads, d_model, d_head, s)
    @w_k = build_head_mats(n_heads, d_model, d_head, s)
    @w_v = build_head_mats(n_heads, d_model, d_head, s)

    @w_o = Mat.new(d_model, d_model)
    @w_o.fill_random(s)

    @w_ff1 = Mat.new(d_model, d_ff)
    @w_ff1.fill_random(s)

    @w_ff2 = Mat.new(d_ff, d_model)
    @w_ff2.fill_random(s)
  end

  def build_head_mats(n, rows, cols, scale)
    seed = Mat.new(rows, cols)
    seed.fill_random(scale)
    out = [seed]
    i = 1
    while i < n
      m = Mat.new(rows, cols)
      m.fill_random(scale)
      out.push(m)
      i += 1
    end
    out
  end
end

# ----------------------------------------------------------------------------
#   Smoke test: build a tiny model state and verify shapes.
# ----------------------------------------------------------------------------

d_model        = 16
d_ff           = 32
n_heads        = 2
d_head         = d_model / n_heads
n_layers       = 2
context_length = 8
vocab_size     = 7

s = 1.0 / Math.sqrt(d_model)

token_embed = Mat.new(vocab_size, d_model)
token_embed.fill_random(s)

pos_embed = Mat.new(context_length, d_model)
pos_embed.fill_random(s)

# Stack of blocks: [seed] + push
blocks = [Block.new(d_model, d_head, d_ff, n_heads)]
li = 1
while li < n_layers
  blocks.push(Block.new(d_model, d_head, d_ff, n_heads))
  li += 1
end

lm_head = Mat.new(d_model, vocab_size)
lm_head.fill_random(s)

puts "Built tiny transformer state"
puts "  blocks=" + blocks.length.to_s
puts "  block[0] w_q heads=" + blocks[0].w_q.length.to_s
puts "  token_embed shape=" + token_embed.nrows.to_s + "x" + token_embed.ncols.to_s
puts "  lm_head shape="     + lm_head.nrows.to_s     + "x" + lm_head.ncols.to_s

# Single matmul: (T × d_model) · (d_model × vocab_size) → (T × vocab_size)
x = Mat.new(3, d_model)
x.fill_random(s)
logits = x.matmul(lm_head)
puts "  logits shape=" + logits.nrows.to_s + "x" + logits.ncols.to_s
