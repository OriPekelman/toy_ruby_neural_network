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
  # Same shape as Toy::GPT2Config but with the llama-extra fields. Kept
  # as a separate class so Spinel's field-name lookups don't have to
  # disambiguate (e.g. `cfg.n_kv` exists here, not on GPT2Config).
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
  #
  # Field names l_attn / l_ffn (not attn / ffn) so Spinel doesn't
  # collapse them with Toy::GPT2Block#attn (CausalSelfAttention) and
  # Toy::GPT2Block#ffn (FFN) — different concrete types.
  class SmolLM2Block
    attr_accessor :rn1, :rn2, :l_attn, :l_ffn

    def initialize(lcfg, rope_obj)
      @rn1    = Toy::RMSNorm.new(lcfg.d_model)
      @rn1.eps = lcfg.rms_eps
      @rn2    = Toy::RMSNorm.new(lcfg.d_model)
      @rn2.eps = lcfg.rms_eps
      @l_attn = Toy::GQAttention.new(lcfg.d_model, lcfg.n_heads, lcfg.n_kv, rope_obj)
      @l_ffn  = Toy::SwiGLU.new(lcfg.d_model, lcfg.d_ff)
    end

    # x: [T, D] → [T, D].  pos_start: absolute position of row 0 of x.
    def forward(x, pos_start)
      x.add!(@l_attn.forward(@rn1.forward(x), pos_start))   # residual after attention
      x.add!(@l_ffn.forward(@rn2.forward(x)))               # residual after FFN
      x
    end

    def param_count
      @rn1.param_count + @rn2.param_count +
        @l_attn.param_count + @l_ffn.param_count
    end
  end

  # SmolLM2 / generic llama-family decoder LM.
  #
  # Field-name notes (Spinel #537 — field-name collapse across classes):
  #   Toy::GPT2 already owns `cfg`, `stack`, `final_norm` accessors
  #   that return GPT-2 types. SmolLM2's versions return *different*
  #   types and would collapse. So every field on this class is
  #   `l_*` prefixed (l for llama-family). One-letter cost, no
  #   collapses anywhere downstream.
  class SmolLM2
    attr_accessor :l_cfg, :l_token_embed, :l_final_norm, :l_stack, :l_rope

    def initialize(lcfg)
      @l_cfg         = lcfg
      @l_token_embed = Toy::Embedding.new(lcfg.vocab, lcfg.d_model)
      @l_final_norm  = Toy::RMSNorm.new(lcfg.d_model)
      @l_final_norm.eps = lcfg.rms_eps
      @l_rope        = Toy::RoPE.new(lcfg.d_model / lcfg.n_heads,
                                     lcfg.ctx, lcfg.rope_base)

      @l_stack = [Toy::SmolLM2Block.new(lcfg, @l_rope)]
      li = 1
      while li < lcfg.n_layers
        @l_stack.push(Toy::SmolLM2Block.new(lcfg, @l_rope))
        li += 1
      end
    end

    # ids: Array<Int> (length T), pos_start: Int → logits [T, V]
    def forward(ids, pos_start)
      x = @l_token_embed.lookup(ids)                         # [T, D]
      li = 0
      while li < @l_cfg.n_layers
        x = @l_stack[li].forward(x, pos_start)               # [T, D]
        li += 1
      end
      x_final = @l_final_norm.forward(x)                     # [T, D]
      x_final.matmul_t(@l_token_embed.weight)                # [T, V]  (tied)
    end

    # Total trainable parameter count (tied embeddings counted once).
    def param_count
      total = @l_token_embed.param_count + @l_final_norm.param_count
      li = 0
      while li < @l_cfg.n_layers
        total = total + @l_stack[li].param_count
        li += 1
      end
      total
    end

    # Build a multi-line description of the architecture and return
    # it as a String. Caller does `puts model.describe`.
    def describe
      sblk0 = @l_stack[0]
      s = "Toy::SmolLM2 (" + Toy.fmt_count(param_count) + " params)\n"
      s = s + "  config: vocab=" + @l_cfg.vocab.to_s
      s = s + " d_model=" + @l_cfg.d_model.to_s
      s = s + " n_heads=" + @l_cfg.n_heads.to_s
      s = s + " n_kv=" + @l_cfg.n_kv.to_s
      s = s + " d_ff=" + @l_cfg.d_ff.to_s
      s = s + " n_layers=" + @l_cfg.n_layers.to_s
      s = s + " ctx=" + @l_cfg.ctx.to_s
      s = s + " rope_base=" + @l_cfg.rope_base.to_s + "\n"
      s = s + "  l_token_embed: " + @l_token_embed.summary
      s = s + "  [" + Toy.fmt_count(@l_token_embed.param_count) + "]\n"
      s = s + "  l_rope:        " + @l_rope.summary + "\n"
      s = s + "  l_stack: " + @l_cfg.n_layers.to_s + " × SmolLM2Block\n"
      s = s + "    rn1:    " + sblk0.rn1.summary + "\n"
      s = s + "    l_attn: " + sblk0.l_attn.summary + "\n"
      s = s + "    rn2:    " + sblk0.rn2.summary + "\n"
      s = s + "    l_ffn:  " + sblk0.l_ffn.summary + "\n"
      s = s + "    (per-block params: " + Toy.fmt_count(sblk0.param_count) + ")\n"
      s = s + "  l_final_norm: " + @l_final_norm.summary + "\n"
      s = s + "  unembed: tied to l_token_embed (logits = x · l_token_embed.T)"
      s
    end
  end
end
