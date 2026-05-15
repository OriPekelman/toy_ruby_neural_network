# lib/toy_smollm2_loader.rb — GGUF weight load + config reader for Toy::SmolLM2.
#
# Kept separate from lib/gguf_load.rb so a demo that only uses
# GPT-2 doesn't have to pull SmolLM2 types into Spinel's compile graph.

require_relative "gguf_load"
require_relative "toy_smollm2"

module GGUFLoad
  # Llama-family weight load into a Toy::SmolLM2.
  #
  # Tensor name conventions match prep/convert_smollm2_to_gguf.py.
  # The converter has already transposed every nn.Linear weight from
  # HF's [out, in] to our [in, out] orientation.
  def self.load_toy_smollm2(llm, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " (" + n_tensors.to_s + " tensors)"

    lcfg    = llm.l_cfg
    d_model = lcfg.d_model
    n_heads = lcfg.n_heads
    n_kv    = lcfg.n_kv
    d_head  = d_model / n_heads

    read_mat(handle,   "token_embd.weight",  llm.l_token_embed.weight, n_tensors)
    read_array(handle, "output_norm.weight", llm.l_final_norm.gamma,   n_tensors)

    li = 0
    while li < lcfg.n_layers
      # Spinel name-collapse defense: `sblk` is unique across the program.
      sblk   = llm.l_stack[li]
      prefix = "blk." + li.to_s

      read_array(handle, prefix + ".attn_norm.weight", sblk.rn1.gamma, n_tensors)
      read_array(handle, prefix + ".ffn_norm.weight",  sblk.rn2.gamma, n_tensors)

      # Q: full [d_model, n_heads * d_head] = [d_model, d_model]
      read_split_heads_weight(handle, prefix + ".attn_q.weight",
                               sblk.l_attn.w_q, n_heads, d_model, d_head, n_tensors)
      # K, V: narrower [d_model, n_kv * d_head] — uses the GQA reader.
      read_split_kv_weight(handle, prefix + ".attn_k.weight",
                            sblk.l_attn.w_k, n_kv, d_model, d_head, n_tensors)
      read_split_kv_weight(handle, prefix + ".attn_v.weight",
                            sblk.l_attn.w_v, n_kv, d_model, d_head, n_tensors)
      read_mat(handle,   prefix + ".attn_output.weight", sblk.l_attn.w_o, n_tensors)

      read_mat(handle,   prefix + ".ffn_gate.weight", sblk.l_ffn.w_gate, n_tensors)
      read_mat(handle,   prefix + ".ffn_up.weight",   sblk.l_ffn.w_up,   n_tensors)
      read_mat(handle,   prefix + ".ffn_down.weight", sblk.l_ffn.w_down, n_tensors)

      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end
end

# Read llama-family hyperparameters from a GGUF's kv metadata. Mirrors
# GPT2ConfigLoader but for `llama.*` keys (set by convert_smollm2_to_gguf.py).
module SmolLM2ConfigLoader
  def self.read(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "SmolLM2ConfigLoader: failed to open " + path
      return Toy::SmolLM2Config.new(0, 0, 0, 0, 0, 0, 0, 10000.0, 1.0e-5)
    end
    vocab     = TinyNN.tnn_gguf_get_u32(handle, "llama.vocab_size")
    d_model   = TinyNN.tnn_gguf_get_u32(handle, "llama.embedding_length")
    d_ff      = TinyNN.tnn_gguf_get_u32(handle, "llama.feed_forward_length")
    n_head    = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count")
    n_kv      = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count_kv")
    n_layer   = TinyNN.tnn_gguf_get_u32(handle, "llama.block_count")
    ctx       = TinyNN.tnn_gguf_get_u32(handle, "llama.context_length")
    rope_base = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.freq_base")
    rms_eps   = TinyNN.tnn_gguf_get_f32(handle, "llama.attention.layer_norm_rms_epsilon")
    TinyNN.tnn_gguf_free(handle)
    Toy::SmolLM2Config.new(vocab, d_model, n_head, n_kv, d_ff, n_layer,
                           ctx, rope_base, rms_eps)
  end
end
