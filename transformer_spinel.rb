# Spinel-compatible decoder-only Transformer LM.
#
# Same architecture as transformer.rb (multi-head attention, RMSNorm,
# residuals, FFN, cross-entropy, Adam, KV cache) but with storage layouts
# that Spinel's type inference handles cleanly:
#
#   • Mat is a class with flat 1D Float storage. Spinel cannot currently
#     infer "Array of FloatArray" from `Array.new(n) { Array.new(m, 0.0) }`,
#     so all 2D matrices are flat under the hood.
#   • Block / cache / KV cache structures are classes, not hashes —
#     polymorphic hashes with mixed value types confuse Spinel's codegen.
#   • Arrays of objects are built with `[seed] + push` instead of
#     `Array.new(n) { obj }` (Spinel takes an IntArray-only fast path).
#   • Field names: nrows/ncols, not rows/cols (the latter triggers a
#     name-collision in Spinel's iterative type inference).
#
# Requires the Spinel patch to infer_ivar_init_type that propagates the
# fill-type from `Array.new(n, 0.0)` into the containing class's struct.

# ============================================================================
#   Mat: 2D float matrix, flat-storage. Indexed as flat[i * ncols + j].
# ============================================================================
class Mat
  attr_accessor :nrows, :ncols, :flat

  def initialize(nrows, ncols)
    @nrows = nrows
    @ncols = ncols
    @flat  = Array.new(nrows * ncols, 0.0)
  end

  def fill_random(scale)
    # Spinel's `rand` (no args) is the C `rand()` returning a large int,
    # not Ruby's [0.0, 1.0) float. Use `rand(N)` which behaves the same
    # in both: an integer in [0, N).
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = (rand(2000).to_f / 1000.0 - 1.0) * scale
      i += 1
    end
  end

  def fill_zero
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = 0.0
      i += 1
    end
  end

  # (m × n) · (n × p) → (m × p) — using local accumulator (faster than
  # repeated indexed writes since each += would re-load the receiver).
  def matmul(other)
    m = @nrows
    n = @ncols
    p = other.ncols
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[i * n + k] * other.flat[k * p + j]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
      end
      i += 1
    end
    out
  end

  # self · otherᵀ where other has the same column count as self.
  def matmul_t(other)
    m = @nrows
    n = @ncols
    p = other.nrows                # other is (p × n) so otherᵀ is (n × p)
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[i * n + k] * other.flat[j * n + k]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
      end
      i += 1
    end
    out
  end

  # selfᵀ · other where self is (n × m) and other is (n × p) → (m × p)
  def t_matmul(other)
    n = @nrows
    m = @ncols
    p = other.ncols
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[k * m + i] * other.flat[k * p + j]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
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

  # Spinel codegen drops `arr.flat[i] += ...` and `arr.flat[i] *= ...`
  # patterns inside while-loops. Use explicit `=` instead.
  def add!(other)
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = @flat[i] + other.flat[i]
      i += 1
    end
    self
  end

  def scale!(s)
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = @flat[i] * s
      i += 1
    end
    self
  end

  def get(i, j)
    @flat[i * @ncols + j]
  end

  def set_at(i, j, v)
    @flat[i * @ncols + j] = v
  end
end

# ============================================================================
#   Block: a transformer block's parameters.
#
#   norm1_gamma, norm2_gamma : Array of Float (length d_model)
#   w_q, w_k, w_v            : Array of Mat (one per head, each d_model × d_head)
#   w_o, w_ff1, w_ff2        : Mat
# ============================================================================
class Block
  attr_accessor :norm1_gamma, :norm2_gamma,
                :w_q, :w_k, :w_v, :w_o, :w_ff1, :w_ff2

  # Zero-initializes everything. Call .fill_random_all(scale) for params,
  # leave as-is for gradients / Adam moments.
  def initialize(d_model, d_head, d_ff, n_heads)
    @norm1_gamma = Array.new(d_model, 1.0)
    @norm2_gamma = Array.new(d_model, 1.0)

    # Per-head matrices: literal-seed pattern so Spinel types as PtrArray of Mat.
    @w_q = [Mat.new(d_model, d_head)]
    @w_k = [Mat.new(d_model, d_head)]
    @w_v = [Mat.new(d_model, d_head)]
    h = 1
    while h < n_heads
      @w_q.push(Mat.new(d_model, d_head))
      @w_k.push(Mat.new(d_model, d_head))
      @w_v.push(Mat.new(d_model, d_head))
      h += 1
    end

    @w_o   = Mat.new(d_model, d_model)
    @w_ff1 = Mat.new(d_model, d_ff)
    @w_ff2 = Mat.new(d_ff,    d_model)
  end

  def fill_random_all(scale)
    h = 0
    while h < @w_q.length
      @w_q[h].fill_random(scale)
      @w_k[h].fill_random(scale)
      @w_v[h].fill_random(scale)
      h += 1
    end
    @w_o.fill_random(scale)
    @w_ff1.fill_random(scale)
    @w_ff2.fill_random(scale)
  end

  def fill_zero
    n = @norm1_gamma.length
    i = 0
    while i < n
      @norm1_gamma[i] = 0.0
      @norm2_gamma[i] = 0.0
      i += 1
    end
    h = 0
    while h < @w_q.length
      @w_q[h].fill_zero
      @w_k[h].fill_zero
      @w_v[h].fill_zero
      h += 1
    end
    @w_o.fill_zero
    @w_ff1.fill_zero
    @w_ff2.fill_zero
  end
end

# ============================================================================
#   Caches — per-sublayer activations needed by backward.
# ============================================================================

class FFCache
  attr_accessor :pre, :hidden
  def initialize(pre, hidden)
    @pre = pre
    @hidden = hidden
  end
end

class HeadCache
  attr_accessor :q, :k, :v, :attn, :head_out
  def initialize(q, k, v, attn, head_out)
    @q = q
    @k = k
    @v = v
    @attn = attn
    @head_out = head_out
  end
end

class AttnCache
  attr_accessor :per_head, :concat
  def initialize(per_head, concat)
    @per_head = per_head
    @concat = concat
  end
end

class LayerCache
  attr_accessor :rms1, :h_norm1, :attn_cache, :x_attn,
                :rms2, :h_norm2, :ff_cache, :x_out
  def initialize
    @rms1 = nil
    @h_norm1 = nil
    @attn_cache = nil
    @x_attn = nil
    @rms2 = nil
    @h_norm2 = nil
    @ff_cache = nil
    @x_out = nil
  end
end

# Full forward cache used by backward.
class ForwardCache
  attr_accessor :token_ids, :x_embed, :layers,
                :x_block_out, :x_final, :rms_final, :logits
  def initialize
    @token_ids = nil
    @x_embed = nil
    @layers = nil
    @x_block_out = nil
    @x_final = nil
    @rms_final = nil
    @logits = nil
  end
end

# Output of cross_entropy_grad: dL/dlogits matrix + scalar loss.
class LossResult
  attr_accessor :dlogits, :loss
  def initialize(dlogits, loss); @dlogits = dlogits; @loss = loss; end
end

# Gradients for the whole model. Structurally a mirror of TransformerLM's
# parameters: same Block-shaped per-layer grads (we reuse ZeroBlock).
class Gradients
  attr_accessor :token_embed, :pos_embed, :lm_head,
                :norm_final_gamma, :blocks, :loss

  def initialize(vocab_size, d_model, d_ff, n_heads, d_head, n_layers, context_length)
    @token_embed = Mat.new(vocab_size, d_model)
    @pos_embed   = Mat.new(context_length, d_model)
    @lm_head     = Mat.new(d_model, vocab_size)
    @norm_final_gamma = Array.new(d_model, 0.0)
    @blocks = [Block.new(d_model, d_head, d_ff, n_heads)]
    li = 1
    while li < n_layers
      @blocks.push(Block.new(d_model, d_head, d_ff, n_heads))
      li += 1
    end
    @loss = 0.0
  end

  def fill_zero
    @token_embed.fill_zero
    @pos_embed.fill_zero
    @lm_head.fill_zero
    n = @norm_final_gamma.length
    i = 0
    while i < n
      @norm_final_gamma[i] = 0.0
      i += 1
    end
    bi = 0
    while bi < @blocks.length
      @blocks[bi].fill_zero
      bi += 1
    end
    @loss = 0.0
  end
end

# Result wrappers — Spinel doesn't reliably handle [Mat, OtherType] tuple
# destructuring (the tuple gets typed as PolyArray and member access boxes
# the values). One small class per return shape keeps things explicit.
class NormResult
  attr_accessor :y, :rms
  def initialize(y, rms); @y = y; @rms = rms; end
end

class AttnResult
  attr_accessor :proj, :cache
  def initialize(proj, cache); @proj = proj; @cache = cache; end
end

class FFResult
  attr_accessor :out, :cache
  def initialize(o, c); @out = o; @cache = c; end
end

class BlockResult
  attr_accessor :x_out, :cache
  def initialize(x_out, cache); @x_out = x_out; @cache = cache; end
end

# ============================================================================
#   TransformerLM
# ============================================================================
class TransformerLM
  attr_accessor :vocab_size, :d_model, :d_ff, :n_heads, :d_head,
                :n_layers, :context_length,
                :token_embed, :pos_embed, :lm_head,
                :norm_final_gamma, :blocks

  def initialize(vocab_size, d_model, d_ff, n_heads, n_layers, context_length)
    @vocab_size     = vocab_size
    @d_model        = d_model
    @d_ff           = d_ff
    @n_heads        = n_heads
    @d_head         = d_model / n_heads
    @n_layers       = n_layers
    @context_length = context_length

    s = 1.0 / Math.sqrt(d_model)

    @token_embed = Mat.new(vocab_size, d_model)
    @token_embed.fill_random(s)

    @pos_embed = Mat.new(context_length, d_model)
    @pos_embed.fill_random(s)

    @lm_head = Mat.new(d_model, vocab_size)
    @lm_head.fill_random(s)

    @norm_final_gamma = Array.new(d_model, 1.0)

    # Inline Block.new in the literal — Spinel's scan_ivars runs before
    # local-variable types are inferred, so storing through a temp would
    # mistype @blocks's element class.
    @blocks = [Block.new(d_model, @d_head, d_ff, n_heads)]
    @blocks[0].fill_random_all(s)
    li = 1
    while li < n_layers
      @blocks.push(Block.new(d_model, @d_head, d_ff, n_heads))
      @blocks[li].fill_random_all(s)
      li += 1
    end

    # Pre-allocate layer caches so the array's element type is fixed at
    # construction time. Forward populates fields on these existing objects.
    @layer_caches = [LayerCache.new]
    li = 1
    while li < n_layers
      @layer_caches.push(LayerCache.new)
      li += 1
    end
  end

  attr_accessor :layer_caches

  # ----- Forward -----

  # x[i] = token_embed[token_ids[i]] + pos_embed[start_pos + i]
  def embed(token_ids, start_pos)
    t = token_ids.length
    out = Mat.new(t, @d_model)
    i = 0
    while i < t
      tok_id = token_ids[i]
      j = 0
      while j < @d_model
        out.flat[i * @d_model + j] =
          @token_embed.flat[tok_id * @d_model + j] +
          @pos_embed.flat[(start_pos + i) * @d_model + j]
        j += 1
      end
      i += 1
    end
    out
  end

  # RMSNorm: y_j = gamma_j * x_j / sqrt(mean(x²) + eps),  per row.
  # Returns a NormResult holding the normed Mat and the per-row rms.
  def rms_norm(x, gamma)
    eps = 1.0e-5
    d = gamma.length
    t = x.nrows
    rms = Array.new(t, 0.0)
    out = Mat.new(t, d)

    i = 0
    while i < t
      sumsq = 0.0
      j = 0
      while j < d
        v = x.flat[i * d + j]
        sumsq += v * v
        j += 1
      end
      r = Math.sqrt(sumsq / d + eps)
      rms[i] = r
      j = 0
      while j < d
        out.flat[i * d + j] = x.flat[i * d + j] * gamma[j] / r
        j += 1
      end
      i += 1
    end

    NormResult.new(out, rms)
  end

  # Row-wise softmax with numerical-stability max-shift, in place on `m`.
  def softmax_rows!(m)
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
        sum += e
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

  # Causal mask: for each row i, set scores[i, j] = -1e30 for j > query_offset + i.
  def apply_causal_mask!(scores, query_offset)
    t = scores.nrows
    n = scores.ncols
    i = 0
    while i < t
      first_masked = query_offset + i + 1
      j = first_masked
      while j < n
        scores.flat[i * n + j] = -1.0e30
        j += 1
      end
      i += 1
    end
  end

  # Concatenate per-head outputs side by side: n_heads × (T × d_head) → (T × d_model)
  def hstack_heads(per_head)
    t = per_head[0].head_out.nrows
    out = Mat.new(t, @d_model)
    h = 0
    while h < @n_heads
      head = per_head[h].head_out
      base = h * @d_head
      i = 0
      while i < t
        j = 0
        while j < @d_head
          out.flat[i * @d_model + (base + j)] = head.flat[i * @d_head + j]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end

  # Multi-head self-attention. Returns AttnResult.
  def self_attention(h_in, block)
    # Force h_in's type inference via an early Mat-typed access.
    t_seq = h_in.nrows
    inv_sqrt = 1.0 / Math.sqrt(@d_head)

    # Build per-head caches with the seed-then-push pattern.
    head0 = self_attention_head(h_in, block, 0, inv_sqrt)
    per_head = [head0]
    hi = 1
    while hi < @n_heads
      per_head.push(self_attention_head(h_in, block, hi, inv_sqrt))
      hi += 1
    end

    concat = hstack_heads(per_head)
    proj   = concat.matmul(block.w_o)

    AttnResult.new(proj, AttnCache.new(per_head, concat))
  end

  def self_attention_head(h_in, block, head_idx, inv_sqrt)
    q = h_in.matmul(block.w_q[head_idx])
    k = h_in.matmul(block.w_k[head_idx])
    v = h_in.matmul(block.w_v[head_idx])

    # scores = (Q · Kᵀ) / sqrt(d_head)
    scores = q.matmul_t(k)
    scores.scale!(inv_sqrt)
    apply_causal_mask!(scores, 0)

    softmax_rows!(scores)
    head_out = scores.matmul(v)

    HeadCache.new(q, k, v, scores, head_out)
  end

  # FFN: relu(h · W_ff1) · W_ff2.  Returns (out_mat, FFCache).
  def feed_forward(h, block)
    pre = h.matmul(block.w_ff1)
    hidden = Mat.new(pre.nrows, pre.ncols)
    n = pre.nrows * pre.ncols
    i = 0
    while i < n
      v = pre.flat[i]
      hidden.flat[i] = v > 0.0 ? v : 0.0
      i += 1
    end
    out = hidden.matmul(block.w_ff2)
    FFResult.new(out, FFCache.new(pre, hidden))
  end

  # Full forward pass. Writes intermediates into @layer_caches and @cache,
  # which are pre-allocated so their types are unambiguous to Spinel.
  # Returns the logits Mat (T × vocab_size).
  def forward(token_ids)
    cache = ForwardCache.new
    cache.token_ids = token_ids

    x = embed(token_ids, 0)
    cache.x_embed = x

    x_cur = x
    li = 0
    while li < @n_layers
      transformer_block_into(x_cur, @blocks[li], @layer_caches[li])
      x_cur = @layer_caches[li].x_out
      li += 1
    end
    cache.layers = @layer_caches
    cache.x_block_out = x_cur

    nr = rms_norm(x_cur, @norm_final_gamma)
    cache.x_final   = nr.y
    cache.rms_final = nr.rms

    cache.logits = nr.y.matmul(@lm_head)
    @cache = cache
    cache.logits
  end

  attr_accessor :cache

  # Same as transformer_block but writes into a pre-existing LayerCache.
  def transformer_block_into(x, block, cache)
    nr1 = rms_norm(x, block.norm1_gamma)
    h1  = nr1.y
    cache.h_norm1 = h1
    cache.rms1    = nr1.rms

    sa = self_attention(h1, block)
    cache.attn_cache = sa.cache
    x_attn = x.add(sa.proj)
    cache.x_attn = x_attn

    nr2 = rms_norm(x_attn, block.norm2_gamma)
    h2  = nr2.y
    cache.h_norm2 = h2
    cache.rms2    = nr2.rms

    ff = feed_forward(h2, block)
    cache.ff_cache = ff.cache
    x_out = x_attn.add(ff.out)
    cache.x_out = x_out
  end

  # Cross-entropy on next-token prediction. dL/dlogits = softmax(logits) - one_hot(target).
  # Loss is averaged over the (T-1) prediction positions.
  def cross_entropy_grad(logits, token_ids)
    n_pred = token_ids.length - 1
    dlogits = Mat.new(logits.nrows, logits.ncols)
    total_loss = 0.0
    if n_pred <= 0
      return LossResult.new(dlogits, 0.0)
    end
    inv_n = 1.0 / n_pred
    v = logits.ncols

    i = 0
    while i < n_pred
      base = i * v
      mx = logits.flat[base]
      j = 1
      while j < v
        val = logits.flat[base + j]
        if val > mx
          mx = val
        end
        j += 1
      end
      sum = 0.0
      j = 0
      while j < v
        e = Math.exp(logits.flat[base + j] - mx)
        sum += e
        j += 1
      end
      target = token_ids[i + 1]
      target_logit = logits.flat[base + target]
      pt = Math.exp(target_logit - mx) / sum
      if pt < 1.0e-12
        pt = 1.0e-12
      end
      total_loss -= Math.log(pt)

      j = 0
      while j < v
        p = Math.exp(logits.flat[base + j] - mx) / sum
        dlogits.flat[base + j] = p * inv_n
        j += 1
      end
      ti = base + target
      dlogits.flat[ti] = dlogits.flat[ti] - inv_n

      i += 1
    end

    LossResult.new(dlogits, total_loss / n_pred)
  end

  # ----- Backward -----
  #
  # Each backward helper writes its parameter-gradients into a target
  # ZeroBlock (or a Mat passed in) and returns the gradient w.r.t. its
  # input as a single Mat. Avoiding tuple returns keeps Spinel's type
  # inference happy.

  # RMSNorm backward.
  #   For y = gamma * x / r,  with r = sqrt(mean(x²) + eps):
  #     dL/dx_k    = (dy_k * gamma_k - x_k * coef) / r,
  #         coef = (Σ_j dy_j * gamma_j * x_j) / (d * r²)
  #     dL/dgamma_j (summed over rows) += dy_j * x_j / r
  # We recompute rms from x rather than taking it as a param. Spinel can't
  # currently propagate FloatArray param types across class-method call
  # sites (its body-usage inference only resolves user-class types), so a
  # `rms` parameter would type as int and the math would silently break.
  def rms_norm_backward(x, gamma, dy, target_dgamma)
    eps = 1.0e-5
    d = gamma.length
    t_seq = x.nrows
    dx = Mat.new(t_seq, d)

    i = 0
    while i < t_seq
      sumsq = 0.0
      j = 0
      while j < d
        v = x.flat[i * d + j]
        sumsq += v * v
        j += 1
      end
      r = Math.sqrt(sumsq / d + eps)

      inner = 0.0
      j = 0
      while j < d
        inner += dy.flat[i * d + j] * gamma[j] * x.flat[i * d + j]
        j += 1
      end
      coef = inner / (d * r * r)

      j = 0
      while j < d
        dx.flat[i * d + j] =
          (dy.flat[i * d + j] * gamma[j] - x.flat[i * d + j] * coef) / r
        target_dgamma[j] = target_dgamma[j] +
                           dy.flat[i * d + j] * x.flat[i * d + j] / r
        j += 1
      end
      i += 1
    end

    dx
  end

  # Row-wise softmax backward (for attention).
  #   d_scores[i,j] = attn[i,j] * (d_attn[i,j] - Σk attn[i,k]·d_attn[i,k])
  def softmax_rows_backward(softmax_out, d_softmax)
    t_seq = softmax_out.nrows
    n = softmax_out.ncols
    out = Mat.new(t_seq, n)
    i = 0
    while i < t_seq
      base = i * n
      s = 0.0
      j = 0
      while j < n
        s += softmax_out.flat[base + j] * d_softmax.flat[base + j]
        j += 1
      end
      j = 0
      while j < n
        out.flat[base + j] = softmax_out.flat[base + j] *
                              (d_softmax.flat[base + j] - s)
        j += 1
      end
      i += 1
    end
    out
  end

  # Split a (T × d_model) matrix back into n_heads × (T × d_head) heads.
  def hsplit_heads(d_concat)
    t_seq = d_concat.nrows
    out = [Mat.new(t_seq, @d_head)]
    h = 1
    while h < @n_heads
      out.push(Mat.new(t_seq, @d_head))
      h += 1
    end
    h = 0
    while h < @n_heads
      base = h * @d_head
      m = out[h]
      i = 0
      while i < t_seq
        j = 0
        while j < @d_head
          m.flat[i * @d_head + j] = d_concat.flat[i * @d_model + (base + j)]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end

  # FFN backward. Writes w_ff1, w_ff2 grads into target_block. Returns d_h.
  def feed_forward_backward(d_ff_out, h, ff_cache, block, target_block)
    t_seq = h.nrows           # type hint: h is a Mat
    d_w_ff2  = ff_cache.hidden.t_matmul(d_ff_out)
    d_hidden = d_ff_out.matmul_t(block.w_ff2)

    # ReLU': zero where pre-activation was non-positive.
    d_pre = Mat.new(d_hidden.nrows, d_hidden.ncols)
    n = d_hidden.nrows * d_hidden.ncols
    i = 0
    while i < n
      v = ff_cache.pre.flat[i]
      d_pre.flat[i] = v > 0.0 ? d_hidden.flat[i] : 0.0
      i += 1
    end

    d_w_ff1 = h.t_matmul(d_pre)
    d_h     = d_pre.matmul_t(block.w_ff1)

    target_block.w_ff1 = d_w_ff1
    target_block.w_ff2 = d_w_ff2
    d_h
  end

  # Self-attention backward. Writes per-head w_q/k/v + w_o grads into
  # target_block. Returns d_h_in.
  def self_attention_backward(d_proj, h_in, attn_cache, block, target_block)
    t_seq = h_in.nrows        # type hint
    inv_sqrt = 1.0 / Math.sqrt(@d_head)

    # proj = concat · w_o
    d_w_o = attn_cache.concat.t_matmul(d_proj)
    d_concat = d_proj.matmul_t(block.w_o)
    target_block.w_o = d_w_o

    d_outs = self.hsplit_heads(d_concat)

    # Build per-head Q/K/V grads (Mat per head). Seed-then-push for typing.
    d_w_q_heads = [Mat.new(@d_model, @d_head)]
    d_w_k_heads = [Mat.new(@d_model, @d_head)]
    d_w_v_heads = [Mat.new(@d_model, @d_head)]
    h = 1
    while h < @n_heads
      d_w_q_heads.push(Mat.new(@d_model, @d_head))
      d_w_k_heads.push(Mat.new(@d_model, @d_head))
      d_w_v_heads.push(Mat.new(@d_model, @d_head))
      h += 1
    end

    d_h_in = Mat.new(t_seq, @d_model)

    h = 0
    while h < @n_heads
      head = attn_cache.per_head[h]
      d_out_h = d_outs[h]

      # out = attn · V
      d_attn = d_out_h.matmul_t(head.v)
      d_v    = head.attn.t_matmul(d_out_h)

      # softmax row-wise (masked entries had attn = 0 so contribute nothing)
      d_scores = self.softmax_rows_backward(head.attn, d_attn)
      d_scores.scale!(inv_sqrt)

      # scores = Q · Kᵀ
      d_q = d_scores.matmul(head.k)
      d_k = d_scores.transpose.matmul(head.q)

      d_w_q_heads[h] = h_in.t_matmul(d_q)
      d_w_k_heads[h] = h_in.t_matmul(d_k)
      d_w_v_heads[h] = h_in.t_matmul(d_v)

      d_h_in.add!(d_q.matmul_t(block.w_q[h]))
      d_h_in.add!(d_k.matmul_t(block.w_k[h]))
      d_h_in.add!(d_v.matmul_t(block.w_v[h]))

      h += 1
    end

    target_block.w_q = d_w_q_heads
    target_block.w_k = d_w_k_heads
    target_block.w_v = d_w_v_heads
    d_h_in
  end

  # Backward through one block. Writes grads into target_block_grads.
  # Returns d_x_in (Mat).
  def transformer_block_backward(dx_out, x_in, block, layer_cache, target_block_grads)
    # x_in is only passed as an arg below — never accessed directly. Spinel's
    # body-usage parameter inference needs at least one method call to type
    # the param. `.nrows` is a Mat-only method, so this anchors x_in's type.
    _x_t = x_in.nrows

    # FFN sublayer residual: x_out = x_attn + ff_out → grad flows to both branches.
    d_h_norm2 = self.feed_forward_backward(dx_out, layer_cache.h_norm2,
                                           layer_cache.ff_cache, block, target_block_grads)
    d_x_attn_via_norm = self.rms_norm_backward(layer_cache.x_attn, block.norm2_gamma,
                                               d_h_norm2, target_block_grads.norm2_gamma)
    d_x_attn = dx_out.add(d_x_attn_via_norm)

    # Attention sublayer residual: x_attn = x_in + attn_proj.
    d_h_norm1 = self.self_attention_backward(d_x_attn, layer_cache.h_norm1,
                                             layer_cache.attn_cache, block, target_block_grads)
    d_x_in_via_norm = self.rms_norm_backward(x_in, block.norm1_gamma,
                                             d_h_norm1, target_block_grads.norm1_gamma)

    d_x_attn.add(d_x_in_via_norm)
  end

  # ----- Optimization -----
  #
  # We use plain SGD here rather than Adam: Adam needs parallel @m and @v
  # shadows of the entire parameter set, and Spinel's parameter-type
  # inference for class methods that take grads-shaped arguments hits
  # multiple of the limitations we've already worked around. Plain SGD
  # demonstrates that gradient accumulation and parameter updates compile
  # and run; Adam is the same structure × 2 with a couple of extra ops.

  def apply_gradients_sgd(grads, lr)
    self.sgd_step_mat(@token_embed, grads.token_embed, lr)
    self.sgd_step_mat(@pos_embed,   grads.pos_embed,   lr)
    self.sgd_step_mat(@lm_head,     grads.lm_head,     lr)
    self.sgd_step_vec(@norm_final_gamma, grads.norm_final_gamma, lr)

    li = 0
    while li < @n_layers
      self.sgd_step_block(@blocks[li], grads.blocks[li], lr)
      li += 1
    end
  end

  def sgd_step_mat(p, g, lr)
    n = p.flat.length
    i = 0
    while i < n
      p.flat[i] = p.flat[i] - lr * g.flat[i]   # not `-=`: Spinel drops it
      i += 1
    end
  end

  def sgd_step_vec(p, g, lr)
    n = p.length
    i = 0
    while i < n
      p[i] = p[i] - lr * g[i]
      i += 1
    end
  end

  def sgd_step_block(p_block, g_block, lr)
    self.sgd_step_vec(p_block.norm1_gamma, g_block.norm1_gamma, lr)
    self.sgd_step_vec(p_block.norm2_gamma, g_block.norm2_gamma, lr)
    self.sgd_step_mat(p_block.w_o,   g_block.w_o,   lr)
    self.sgd_step_mat(p_block.w_ff1, g_block.w_ff1, lr)
    self.sgd_step_mat(p_block.w_ff2, g_block.w_ff2, lr)

    h = 0
    while h < @n_heads
      self.sgd_step_mat(p_block.w_q[h], g_block.w_q[h], lr)
      self.sgd_step_mat(p_block.w_k[h], g_block.w_k[h], lr)
      self.sgd_step_mat(p_block.w_v[h], g_block.w_v[h], lr)
      h += 1
    end
  end

  # One training step: forward, backward, apply.
  def train_step(token_ids, grads, lr)
    grads.fill_zero
    self.forward(token_ids)
    self.backward(token_ids, grads)
    self.apply_gradients_sgd(grads, lr)
    grads.loss
  end

  # Block i's input is the previous block's output, or the embedded input
  # for block 0. Returning a Mat in both branches makes Spinel type the
  # method's return as Mat — useful as an argument to backward helpers
  # whose param inference doesn't trust ternary expressions.
  def x_in_for_layer(li)
    if li == 0
      return @cache.x_embed
    end
    @cache.layers[li - 1].x_out
  end

  # Embedding backward: each row of dx routes to its token's embedding row
  # and to position i's positional embedding row. Repeats accumulate.
  # Workaround: avoid `arr.flat[i] += x` (Spinel-codegen drops it).
  def embed_backward(token_ids, dx, target_grads)
    t_seq = token_ids.length
    i = 0
    while i < t_seq
      tok = token_ids[i]
      j = 0
      while j < @d_model
        pi = i * @d_model + j
        target_grads.token_embed.flat[tok * @d_model + j] += dx.flat[pi]
        target_grads.pos_embed.flat[pi]                   += dx.flat[pi]
        j += 1
      end
      i += 1
    end
  end

  # Full backward pass. Fills `target_grads` with this example's gradients
  # and the loss. Caller is responsible for calling forward(token_ids) first.
  def backward(token_ids, target_grads)
    n_pred = token_ids.length - 1
    if n_pred <= 0
      target_grads.loss = 0.0
      return
    end

    lr = self.cross_entropy_grad(@cache.logits, token_ids)
    target_grads.loss = lr.loss

    # LM head: logits = x_final · lm_head
    target_grads.lm_head = @cache.x_final.t_matmul(lr.dlogits)
    dx_final = lr.dlogits.matmul_t(@lm_head)

    # Final RMSNorm. Use `self.` so Spinel's call-site parameter inference
    # picks up the typed args (only fires for explicit-receiver calls).
    dx = self.rms_norm_backward(@cache.x_block_out, @norm_final_gamma,
                                dx_final, target_grads.norm_final_gamma)

    # Each block in reverse.
    li = @n_layers - 1
    while li >= 0
      dx = self.transformer_block_backward(dx, self.x_in_for_layer(li),
                                           @blocks[li], @cache.layers[li],
                                           target_grads.blocks[li])
      li -= 1
    end

    self.embed_backward(token_ids, dx, target_grads)
  end

  # One transformer block (pre-norm). Returns BlockResult.
  # Locals are explicit so Spinel can type-trace argument types into the
  # called methods (passing `nr1.y` directly through doesn't propagate).
  def transformer_block(x, block)
    cache = LayerCache.new

    nr1 = rms_norm(x, block.norm1_gamma)
    h1  = nr1.y
    cache.h_norm1 = h1
    cache.rms1    = nr1.rms

    sa = self_attention(h1, block)
    cache.attn_cache = sa.cache
    x_attn = x.add(sa.proj)
    cache.x_attn = x_attn

    nr2 = rms_norm(x_attn, block.norm2_gamma)
    h2  = nr2.y
    cache.h_norm2 = h2
    cache.rms2    = nr2.rms

    ff = feed_forward(h2, block)
    cache.ff_cache = ff.cache
    x_out = x_attn.add(ff.out)
    cache.x_out = x_out

    BlockResult.new(x_out, cache)
  end
end

# ============================================================================
#   Smoke test: build a model and run embed + rms_norm.
# ============================================================================

model = TransformerLM.new(7, 16, 32, 2, 2, 8)
puts "Built model"
puts "  vocab="           + model.vocab_size.to_s
puts "  d_model="         + model.d_model.to_s
puts "  blocks="          + model.blocks.length.to_s
puts "  heads/block="     + model.blocks[0].w_q.length.to_s

token_ids = [0, 1, 2]

logits = model.forward(token_ids)
puts "  logits shape="    + logits.nrows.to_s + "x" + logits.ncols.to_s
puts "  logits.flat[0]="
puts logits.flat[0]
puts "  logits.flat[1]="
puts logits.flat[1]
puts "  logits.flat[6]="
puts logits.flat[6]
puts "  x_final.flat[0]="
puts model.cache.x_final.flat[0]
puts "  lm_head.flat[0]="
puts model.lm_head.flat[0]
puts "  lm_head.flat[7]="
puts model.lm_head.flat[7]

lr = model.cross_entropy_grad(logits, token_ids)
puts "  loss="
puts lr.loss
puts "  dlogits shape="   + lr.dlogits.nrows.to_s + "x" + lr.dlogits.ncols.to_s
puts "  dlogits.flat[0]="
puts lr.dlogits.flat[0]
puts "  dlogits.flat[6]="
puts lr.dlogits.flat[6]

# Backward pass: fill grads.
grads = Gradients.new(7, 16, 32, 2, 8, 2, 8)
model.backward(token_ids, grads)
puts "  grad token_embed shape=" + grads.token_embed.nrows.to_s + "x" + grads.token_embed.ncols.to_s
puts "  grad lm_head shape="     + grads.lm_head.nrows.to_s + "x" + grads.lm_head.ncols.to_s
puts "  grads.lm_head.flat[0] (after first backward):"
puts grads.lm_head.flat[0]
puts "  grads.token_embed.flat[0]:"
puts grads.token_embed.flat[0]
puts "  loss before training="
puts grads.loss

# Training loop on a tiny corpus.
seqs_a = [0, 1, 2]
seqs_b = [3, 4, 5]
seqs_c = [1, 2, 3]
puts ""
puts "Training (40 steps over 3 sequences, SGD lr=0.05):"
step = 0
while step < 40
  total_loss = 0.0
  total_loss += model.train_step(seqs_a, grads, 0.05)
  total_loss += model.train_step(seqs_b, grads, 0.05)
  total_loss += model.train_step(seqs_c, grads, 0.05)
  if step % 4 == 0
    puts "  step="
    puts step
    puts "  mean_loss="
    puts total_loss / 3.0
  end
  step += 1
end

