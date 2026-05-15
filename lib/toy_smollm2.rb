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
  class SmolLM2Block
    attr_accessor :rn1, :rn2, :attn, :ffn

    def initialize(cfg, rope_obj)
      @rn1  = Toy::RMSNorm.new(cfg.d_model)
      @rn1.eps = cfg.rms_eps
      @rn2  = Toy::RMSNorm.new(cfg.d_model)
      @rn2.eps = cfg.rms_eps
      @attn = Toy::GQAttention.new(cfg.d_model, cfg.n_heads, cfg.n_kv, rope_obj)
      @ffn  = Toy::SwiGLU.new(cfg.d_model, cfg.d_ff)
    end

    # x: [T, D] → [T, D].  pos_start: absolute position of row 0 of x.
    def forward(x, pos_start)
      x.add!(@attn.forward(@rn1.forward(x), pos_start))   # residual after attention
      x.add!(@ffn.forward(@rn2.forward(x)))               # residual after FFN
      x
    end
  end

  # SmolLM2 / generic llama-family decoder LM.
  #
  # Field name notes:
  #   - block stack is `stack` (same as Toy::GPT2; avoids collapse with
  #     TransformerLM#blocks → spinel#537).
  #   - `rope` holds the shared RoPE table — one per model.
  class SmolLM2
    attr_accessor :cfg, :token_embed, :final_norm, :stack, :rope

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
      x_final.matmul_t(@token_embed.weight)                  # [T, V]  (tied)
    end
  end
end
