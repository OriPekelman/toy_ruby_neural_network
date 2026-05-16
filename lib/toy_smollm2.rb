# lib/toy_smollm2.rb — Toy::SmolLM2: llama-family decoder LM.
#
# Same block shape as Toy::GPT2 (pre-norm, two sublayers, residual on
# each) but with the llama-family substitutions:
#   - RMSNorm instead of LayerNorm
#   - GQAttention (grouped-query) instead of plain MHA
#   - SwiGLU instead of GeLU FFN
#   - RoPE instead of learned absolute position embeddings
#   - no biases anywhere
#   - tied embeddings (token_embed used as the unembed)
#
# Default config matches HuggingFaceTB/SmolLM2-135M:
#   d_model=576, n_heads=9, n_kv=3, d_ff=1536, n_layers=30,
#   vocab=49152, ctx=8192, rope_base=100000, rms_eps=1e-5

require_relative "toy"

module Toy
  class SmolLM2Config
    attr_accessor :vocab, :d_model, :n_heads, :n_kv, :d_ff,
                  :n_layers, :ctx, :rope_base, :rms_eps

    def initialize(vocab, d_model, n_heads, n_kv, d_ff, n_layers,
                   ctx, rope_base, rms_eps)
      @vocab     = vocab
      @d_model   = d_model
      @n_heads   = n_heads
      @n_kv      = n_kv
      @d_ff      = d_ff
      @n_layers  = n_layers
      @ctx       = ctx
      @rope_base = rope_base
      @rms_eps   = rms_eps
    end

    # Convenience: the default that matches SmolLM2-135M on HF.
    def self.smollm2_135m
      Toy::SmolLM2Config.new(49152, 576, 9, 3, 1536, 30,
                             8192, 100000.0, 1.0e-5)
    end
  end

  # Llama-style block: pre-norm + residual on each sublayer.
  class SmolLM2Block
    attr_accessor :rn1, :rn2, :attn, :ffn

    def initialize(cfg, rope_obj)
      @rn1     = Toy::RMSNorm.new(cfg.d_model)
      @rn1.eps = cfg.rms_eps
      @rn2     = Toy::RMSNorm.new(cfg.d_model)
      @rn2.eps = cfg.rms_eps
      @attn    = Toy::GQAttention.new(cfg.d_model, cfg.n_heads, cfg.n_kv, rope_obj)
      @ffn     = Toy::SwiGLU.new(cfg.d_model, cfg.d_ff)
    end

    # x: [T, D] → [T, D].  pos_start: absolute position of row 0 of x.
    def forward(x, pos_start)
      x.add!(@attn.forward(@rn1.forward(x), pos_start))   # residual after attention
      x.add!(@ffn.forward(@rn2.forward(x)))               # residual after FFN
      x
    end

    def param_count
      @rn1.param_count + @rn2.param_count +
        @attn.param_count + @ffn.param_count
    end

    def algorithm_card
      s =  "Algorithm: SmolLM2Block.forward(x, p_start)\n"
      s = s + "  Input/Output: x ∈ R^{T×D};  p_start ∈ ℕ\n"
      s = s + "  1: x ← x + GQAttn(RMSNorm(x; γ_1, ε), p_start)    ▷ residual; RoPE inside attn\n"
      s = s + "  2: x ← x + SwiGLU(RMSNorm(x; γ_2, ε))             ▷ residual\n"
      s = s + "  3: return x"
      s
    end
  end

  # SmolLM2 / generic llama-family decoder LM.
  #
  # Supports both tied and untied output embeddings:
  #   - SmolLM2 / Qwen2.5 / Gemma: tied (logits = x · token_embed.T)
  #   - TinyLlama / Llama-2/3 / Mistral: untied (logits = x · lm_head.T)
  #
  # Untied is opt-in via enable_untied_output! after construction.
  # The output_proj weight is stored as [V, D] (matches token_embed
  # layout) so the same matmul_t code path works for both.
  class SmolLM2
    attr_accessor :cfg, :token_embed, :final_norm, :stack, :rope,
                  :output_proj, :has_untied_output

    def initialize(cfg)
      @cfg         = cfg
      @token_embed = Toy::Embedding.new(cfg.vocab, cfg.d_model)
      @final_norm  = Toy::RMSNorm.new(cfg.d_model)
      @final_norm.eps = cfg.rms_eps
      @rope        = Toy::RoPE.new(cfg.d_model / cfg.n_heads,
                                   cfg.ctx, cfg.rope_base)

      @stack = [Toy::SmolLM2Block.new(cfg, @rope)]
      li = 1
      while li < cfg.n_layers
        @stack.push(Toy::SmolLM2Block.new(cfg, @rope))
        li += 1
      end

      # Always allocate the output projection at full [V, D] shape so
      # Spinel sees a stable Mat with known dimensions from the very
      # first reference. Costs vocab*d_model floats of memory even on
      # tied models (a few MB on SmolLM2, 256MB on TinyLlama) — small
      # next to the actual weights and avoids reassign-after-construct
      # surprises in the AOT type model.
      @output_proj       = Mat.new(cfg.vocab, cfg.d_model)
      @has_untied_output = false
    end

    # Called by the GGUF loader when `output.weight` is present. The
    # Mat is already allocated; this just flips the flag so the
    # forward uses it.
    def enable_untied_output!
      @has_untied_output = true
    end

    # ids: Array<Int> (length T), pos_start: Int → logits [T, V]
    def forward(ids, pos_start)
      x = @token_embed.lookup(ids)                           # [T, D]
      li = 0
      while li < @cfg.n_layers
        x = @stack[li].forward(x, pos_start)                 # [T, D]
        li += 1
      end
      x_final = @final_norm.forward(x)                       # [T, D]
      if @has_untied_output
        x_final.matmul_t(@output_proj)                       # [T, V]  (untied)
      else
        x_final.matmul_t(@token_embed.weight)                # [T, V]  (tied)
      end
    end

    # Total trainable parameter count (tied embeddings counted once).
    def param_count
      total = @token_embed.param_count + @final_norm.param_count
      li = 0
      while li < @cfg.n_layers
        total = total + @stack[li].param_count
        li += 1
      end
      total
    end

    # Build a multi-line description of the architecture and return
    # it as a String. Caller does `puts model.describe`.
    def describe
      blk0 = @stack[0]
      s = "Toy::SmolLM2 (" + Toy.fmt_count(param_count) + " params)\n"
      s = s + "  config: vocab=" + @cfg.vocab.to_s
      s = s + " d_model=" + @cfg.d_model.to_s
      s = s + " n_heads=" + @cfg.n_heads.to_s
      s = s + " n_kv=" + @cfg.n_kv.to_s
      s = s + " d_ff=" + @cfg.d_ff.to_s
      s = s + " n_layers=" + @cfg.n_layers.to_s
      s = s + " ctx=" + @cfg.ctx.to_s
      s = s + " rope_base=" + @cfg.rope_base.to_s + "\n"
      s = s + "  token_embed: " + @token_embed.summary
      s = s + "  [" + Toy.fmt_count(@token_embed.param_count) + "]\n"
      s = s + "  rope:        " + @rope.summary + "\n"
      s = s + "  stack: " + @cfg.n_layers.to_s + " × SmolLM2Block\n"
      s = s + "    rn1:  " + blk0.rn1.summary + "\n"
      s = s + "    attn: " + blk0.attn.summary + "\n"
      s = s + "    rn2:  " + blk0.rn2.summary + "\n"
      s = s + "    ffn:  " + blk0.ffn.summary + "\n"
      s = s + "    (per-block params: " + Toy.fmt_count(blk0.param_count) + ")\n"
      s = s + "  final_norm: " + @final_norm.summary + "\n"
      s = s + "  unembed: tied to token_embed (logits = x · token_embed.T)"
      s
    end

    # Phuong–Hutter style algorithm card. Reads like the paper —
    # tensor shapes annotated on the right, ←  for assignment, ▷ for
    # commentary. See arXiv:2207.09238 §4 for the canonical form.
    def algorithm_card
      unembed_line = @has_untied_output ?
        "  7: P ← e · W_out^⊤                                                  P ∈ R^{T×V}  (untied)" :
        "  7: P ← e · W_e^⊤                                                    P ∈ R^{T×V}  (tied)"
      s =  "Algorithm: Toy::SmolLM2.forward(x, p_start)              [Llama-family decoder]\n"
      s = s + "  Input:    x ∈ {1..V}^T   (token IDs)\n"
      s = s + "            p_start ∈ ℕ    (absolute position of x[0]; for RoPE)\n"
      s = s + "  Output:   P ∈ R^{T×V}    (logits)\n"
      s = s + "  Hyper:    V=" + @cfg.vocab.to_s + " D=" + @cfg.d_model.to_s +
              " H=" + @cfg.n_heads.to_s + " H_kv=" + @cfg.n_kv.to_s +
              " D_f=" + @cfg.d_ff.to_s + " N=" + @cfg.n_layers.to_s +
              " ctx=" + @cfg.ctx.to_s + " θ_base=" + @cfg.rope_base.to_s + "\n"
      s = s + "  Param:    W_e ∈ R^{V×D}                              (token embeddings)\n"
      if @has_untied_output
        s = s + "            W_out ∈ R^{V×D}                            (separate lm_head)\n"
      end
      s = s + "            θ_block_ℓ for ℓ=1..N                       (per-block; see SmolLM2Block)\n"
      s = s + "            γ_f ∈ R^D                                  (final RMSNorm)\n"
      s = s + "            (total " + Toy.fmt_count(param_count) + ")\n"
      s = s + "  1: e ← W_e[x]                                                        e ∈ R^{T×D}\n"
      s = s + "  2: for ℓ ← 1, …, N do\n"
      s = s + "  3:    e ← e + GQAttn(RMSNorm(e; γ_ℓ^1, ε), p_start; θ_ℓ^attn)         e ∈ R^{T×D}\n"
      s = s + "  4:    e ← e + SwiGLU(RMSNorm(e; γ_ℓ^2, ε); θ_ℓ^ffn)                    e ∈ R^{T×D}\n"
      s = s + "  5: end for\n"
      s = s + "  6: e ← RMSNorm(e; γ_f, ε)                                              e ∈ R^{T×D}\n"
      s = s + unembed_line + "\n"
      s = s + "  8: return P"
      s
    end

    # Recursive card: top-level forward + block + every sub-op
    # (RMSNorm, GQAttention, RoPE, SwiGLU) inlined. Useful for the
    # "full pseudocode" view; the top-level alone is the "section-1
    # overview" view.
    def algorithm_card_full
      blk = @stack[0]
      s = algorithm_card + "\n\n"
      s = s + "─── sub-algorithms ─────────────────────────────────────────────────────\n\n"
      s = s + blk.algorithm_card    + "\n\n"
      s = s + blk.rn1.algorithm_card  + "\n\n"
      s = s + blk.attn.algorithm_card + "\n\n"
      s = s + @rope.algorithm_card    + "\n\n"
      s = s + blk.ffn.algorithm_card
      s
    end
  end
end
