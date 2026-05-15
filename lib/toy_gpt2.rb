# lib/toy_gpt2.rb — Toy::GPT2: HF-shape GPT-2 in the sugar API.
#
# The whole model is the three lines of `forward` at the bottom of this
# file. Everything above is configuration + a `GPT2Block` whose `call`
# is two lines.
#
# ┌─ Architecture (same as HF transformers.GPT2Model) ──────────────┐
# │   ids  →  embed (token + position)  →  N × GPT2Block            │
# │        →  final LayerNorm  →  unembed (tied to token_embed.T)   │
# │                                                                  │
# │   GPT2Block (pre-norm):                                          │
# │     x ← x + attn(ln1(x))                                         │
# │     x ← x + ffn (ln2(x))                                         │
# └──────────────────────────────────────────────────────────────────┘

require_relative "toy"

module Toy
  # Hyperparameter bag. Use a class (not Hash) so Spinel sees a
  # concrete struct with Int / Float fields.
  class GPT2Config
    attr_accessor :vocab, :d_model, :n_heads, :d_ff, :n_layers, :ctx

    def initialize(vocab, d_model, n_heads, d_ff, n_layers, ctx)
      @vocab    = vocab
      @d_model  = d_model
      @n_heads  = n_heads
      @d_ff     = d_ff
      @n_layers = n_layers
      @ctx      = ctx
    end
  end

  # One transformer block: pre-norm, residual after attention,
  # pre-norm, residual after FFN.
  class GPT2Block
    attr_accessor :ln1, :ln2, :g_attn, :ffn

    def initialize(cfg)
      @ln1  = Toy::LayerNorm.new(cfg.d_model)
      @ln2  = Toy::LayerNorm.new(cfg.d_model)
      @g_attn = Toy::CausalSelfAttention.new(cfg.d_model, cfg.n_heads)
      @ffn  = Toy::FFN.new(cfg.d_model, cfg.d_ff, :gelu_new)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      x.add!(@g_attn.forward(@ln1.forward(x)))    # residual after attention
      x.add!(@ffn.forward(@ln2.forward(x)))     # residual after FFN
      x
    end
  end

  # GPT-2: decoder-only transformer LM.
  #
  # Field name note: the block array is `stack`, not `blocks`. Spinel's
  # whole-program inference unifies field-name lookups across types,
  # and `blocks` is already taken by TransformerLM (Array<Block>) /
  # ForwardCache (cache.layers).
  class GPT2
    attr_accessor :cfg, :token_embed, :pos_embed, :stack, :final_norm

    def initialize(cfg)
      @cfg         = cfg
      @token_embed = Toy::Embedding.new(cfg.vocab, cfg.d_model)
      @pos_embed   = Toy::Embedding.new(cfg.ctx,   cfg.d_model)
      @final_norm  = Toy::LayerNorm.new(cfg.d_model)

      # Block stack: literal-seed + push so Spinel infers
      # PtrArray<Toy::GPT2Block>.
      @stack = [Toy::GPT2Block.new(cfg)]
      li = 1
      while li < cfg.n_layers
        @stack.push(Toy::GPT2Block.new(cfg))
        li += 1
      end
    end

    # ids: Array<Int> (length T), start_pos: Int → logits [T, V]
    def forward(ids, start_pos)
      x = @token_embed.lookup(ids)                              # [T, D]
      x.add!(@pos_embed.slice(start_pos, ids.length))           # [T, D]

      li = 0
      while li < @cfg.n_layers
        x = @stack[li].forward(x)                                   # [T, D]
        li += 1
      end

      x_final = @final_norm.forward(x)                              # [T, D]
      x_final.matmul_t(@token_embed.weight)                      # [T, V]
    end
  end
end
