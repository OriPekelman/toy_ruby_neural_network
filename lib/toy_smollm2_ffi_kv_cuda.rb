# lib/toy_smollm2_ffi_kv_cuda.rb — Toy::SmolLM2 KV-cache decode via ggml FFI.
#
# Mirror of lib/gpt2_ffi_kv_cuda.rb but for the llama-family architecture:
#   - RMSNorm (no beta) instead of LayerNorm
#   - No biases on Q / K / V / O / FFN projections
#   - SwiGLU FFN: down( silu(gate(x)) * up(x) )
#   - RoPE applied to Q and K before the dot product
#   - GQA: K and V are stored per-`n_kv`-head, not per-`n_heads`-head.
#     Each KV head is shared by group_size = n_heads / n_kv query heads.
#
# Per decode step builds a single-position compute graph; K and V at
# the current position are written into persistent per-layer buffers
# via cpy-into-view (same pattern as the GPT-2 cache). Cost per step:
# constant in prompt length.

require_relative "transformer"
require_relative "toy"
require_relative "toy_smollm2"
require_relative "tinynn_cuda"

# Per-block persistent tensors for the SmolLM2 KV cache.
#
# Q is split per query head (n_heads of them).
# K, V, and their persistent buffers are split per KV head (n_kv of them).
class SmolLM2KVBlockFFICuda
  attr_accessor :t_rn1_gamma, :t_rn2_gamma,
                :t_w_q, :t_w_k, :t_w_v, :t_w_o,
                :t_b_q, :t_b_k, :t_b_v,
                :t_w_gate, :t_w_up, :t_w_down,
                :t_K, :t_V

  def initialize
    @t_rn1_gamma = TinyNNCuda.tnn_null_ptr
    @t_rn2_gamma = TinyNNCuda.tnn_null_ptr
    @t_w_q  = [TinyNNCuda.tnn_null_ptr]
    @t_w_k  = [TinyNNCuda.tnn_null_ptr]
    @t_w_v  = [TinyNNCuda.tnn_null_ptr]
    @t_b_q  = [TinyNNCuda.tnn_null_ptr]
    @t_b_k  = [TinyNNCuda.tnn_null_ptr]
    @t_b_v  = [TinyNNCuda.tnn_null_ptr]
    @t_K    = [TinyNNCuda.tnn_null_ptr]
    @t_V    = [TinyNNCuda.tnn_null_ptr]
    @t_w_o    = TinyNNCuda.tnn_null_ptr
    @t_w_gate = TinyNNCuda.tnn_null_ptr
    @t_w_up   = TinyNNCuda.tnn_null_ptr
    @t_w_down = TinyNNCuda.tnn_null_ptr
  end
end

class SmolLM2KVFFICacheCuda
  attr_accessor :sess, :t_token_embed, :t_final_norm_gamma,
                :t_output, :has_untied_output, :has_qkv_bias,
                :kv_blocks_ffi,
                :max_T, :d_model, :d_ff, :n_heads, :n_kv, :d_head,
                :group_size, :n_layers, :vocab_size, :rope_base,
                :rms_eps, :realized,
                # Stub to satisfy Spinel's type unification with
                # SmolLM2KVFFICache (which uses weight_type for Phase 3
                # Q8-stays-Q8). The CUDA path doesn't act on this yet.
                :weight_type

  def initialize
    @realized   = false
    @max_T      = 0
    @d_model    = 0
    @d_ff       = 0
    @n_heads    = 0
    @n_kv       = 0
    @d_head     = 0
    @group_size = 0
    @n_layers   = 0
    @vocab_size = 0
    @rope_base  = 10000.0
    @rms_eps    = 1.0e-5
    @sess               = TinyNNCuda.tnn_null_ptr
    @t_token_embed      = TinyNNCuda.tnn_null_ptr
    @t_final_norm_gamma = TinyNNCuda.tnn_null_ptr
    @t_output           = TinyNNCuda.tnn_null_ptr
    @has_untied_output  = false
    @has_qkv_bias       = false
    @kv_blocks_ffi      = [SmolLM2KVBlockFFICuda.new]
    @weight_type        = 0   # stub; CUDA path doesn't act on it yet
  end

  # Declare every persistent tensor (weights + K/V buffers) and finalize.
  # `untied`: separate `output.weight` (TinyLlama, Llama-2).
  # `qkv_bias`: Q/K/V have learned biases (Qwen2.x).
  def realize_for(max_T, d_model, d_ff, n_heads, n_kv, n_layers,
                  vocab_size, rope_base, rms_eps, untied, qkv_bias)
    @max_T      = max_T
    @d_model    = d_model
    @d_ff       = d_ff
    @n_heads    = n_heads
    @n_kv       = n_kv
    @d_head     = d_model / n_heads
    @group_size = n_heads / n_kv
    @n_layers   = n_layers
    @vocab_size = vocab_size
    @rope_base  = rope_base
    @rms_eps    = rms_eps

    @sess               = TinyNNCuda.tnn_session_new(1)
    @t_token_embed      = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_final_norm_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
    @has_untied_output  = untied
    @has_qkv_bias       = qkv_bias
    if untied
      @t_output = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    end

    @kv_blocks_ffi = [SmolLM2KVBlockFFICuda.new]
    li = 1
    while li < n_layers
      @kv_blocks_ffi.push(SmolLM2KVBlockFFICuda.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @kv_blocks_ffi[li]
      blk.t_rn1_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_rn2_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

      # Q: n_heads per-head matrices of (d_head, d_model) (transposed for ggml).
      blk.t_w_q = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      if qkv_bias
        blk.t_b_q = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
      end
      hq = 1
      while hq < n_heads
        blk.t_w_q.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        if qkv_bias
          blk.t_b_q.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
        end
        hq = hq + 1
      end

      # K, V (and the persistent K/V buffers): n_kv per-head.
      blk.t_w_k = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_K   = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, max_T,  d_head)]
      blk.t_V   = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, max_T)]
      if qkv_bias
        blk.t_b_k = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
        blk.t_b_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1)]
      end
      hkv = 1
      while hkv < n_kv
        blk.t_w_k.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_K.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, max_T,  d_head))
        blk.t_V.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, max_T))
        if qkv_bias
          blk.t_b_k.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
          blk.t_b_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1))
        end
        hkv = hkv + 1
      end

      blk.t_w_o    = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_model)
      blk.t_w_gate = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_ff,    d_model)
      blk.t_w_up   = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_ff,    d_model)
      blk.t_w_down = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_ff)
      li = li + 1
    end

    TinyNNCuda.tnn_finalize_weights(@sess)
    @realized = true
  end

  # Build the compute graph for one decode position.
  def build_decode_step(pos)
    eps     = @rms_eps
    scale   = 1.0 / Math.sqrt(@d_head.to_f)
    d_model = @d_model
    d_head  = @d_head
    max_T   = @max_T
    bytes_d_head = d_head * 4
    bytes_max_T  = max_T * 4

    # Inputs: token id + RoPE position. Both length 1.
    t_token_id  = TinyNNCuda.tnn_input_1d_i32(@sess, 1)
    t_pos       = TinyNNCuda.tnn_input_1d_i32_ctx(@sess, 1)

    t_x = TinyNNCuda.tnn_get_rows(@sess, @t_token_embed, t_token_id)   # ne=[d_model, 1]

    li = 0
    while li < @n_layers
      t_x = build_block_step(t_x, @kv_blocks_ffi[li], t_pos, pos,
                              scale, eps, bytes_d_head, bytes_max_T)
      li = li + 1
    end

    t_x_final  = TinyNNCuda.tnn_rms_norm(@sess, t_x, @t_final_norm_gamma, eps)
    # Logits: untied path matmuls against t_output (lm_head); tied
    # path against t_token_embed. Both tensors are [vocab, d_model],
    # so the matmul shape is identical either way.
    if @has_untied_output
      t_kv_logits = TinyNNCuda.tnn_matmul(@sess, @t_output, t_x_final)
    else
      t_kv_logits = TinyNNCuda.tnn_matmul(@sess, @t_token_embed, t_x_final)
    end
    TinyNNCuda.tnn_set_output(t_kv_logits)
    SmolLM2KVStepResultCuda.new(t_token_id, t_pos, t_kv_logits)
  end

  def build_block_step(t_x, blk, t_pos, pos, scale, eps,
                        bytes_d_head, bytes_max_T)
    t_h = TinyNNCuda.tnn_rms_norm(@sess, t_x, blk.t_rn1_gamma, eps)

    # --- compute K, V for each KV head (n_kv times), rope K, cpy into buffers ---
    hkv = 0
    while hkv < @n_kv
      t_k_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_k[hkv], t_h)         # ne=[d_head, 1]
      if @has_qkv_bias
        t_k_pre = TinyNNCuda.tnn_add(@sess, t_k_raw, blk.t_b_k[hkv])
      else
        t_k_pre = t_k_raw
      end
      t_k_rot = TinyNNCuda.tnn_rope_ext(@sess, t_k_pre, t_pos, @d_head, @rope_base)
      t_v_raw = TinyNNCuda.tnn_matmul(@sess, t_h, blk.t_w_v[hkv])         # ne=[1, d_head]
      if @has_qkv_bias
        t_v_new = TinyNNCuda.tnn_add(@sess, t_v_raw, blk.t_b_v[hkv])
      else
        t_v_new = t_v_raw
      end

      t_K_slot = TinyNNCuda.tnn_view_2d(@sess, blk.t_K[hkv],
                                      @d_head, 1, bytes_d_head, pos * bytes_d_head)
      t_cpy_k = TinyNNCuda.tnn_cpy(@sess, t_k_rot, t_K_slot)
      t_V_slot = TinyNNCuda.tnn_view_2d(@sess, blk.t_V[hkv],
                                      1, @d_head, bytes_max_T, pos * 4)
      t_cpy_v = TinyNNCuda.tnn_cpy(@sess, t_v_new, t_V_slot)
      TinyNNCuda.tnn_add_to_graph(@sess, t_cpy_k)
      TinyNNCuda.tnn_add_to_graph(@sess, t_cpy_v)
      hkv = hkv + 1
    end

    # --- per-Q-head attention ---
    t_head_out0 = build_attention_qhead_step(t_h, blk, 0, t_pos, pos,
                                              scale, bytes_d_head, bytes_max_T)
    t_head_outs = [t_head_out0]
    hq = 1
    while hq < @n_heads
      t_head_outs.push(build_attention_qhead_step(t_h, blk, hq, t_pos, pos,
                                                    scale, bytes_d_head, bytes_max_T))
      hq = hq + 1
    end

    t_concat = t_head_outs[0]
    hq = 1
    while hq < @n_heads
      t_concat = TinyNNCuda.tnn_concat(@sess, t_concat, t_head_outs[hq], 0)
      hq = hq + 1
    end

    t_out_proj = TinyNNCuda.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_x_attn   = TinyNNCuda.tnn_add(@sess, t_x, t_out_proj)

    # --- SwiGLU FFN ---
    t_h2     = TinyNNCuda.tnn_rms_norm(@sess, t_x_attn, blk.t_rn2_gamma, eps)
    t_gate   = TinyNNCuda.tnn_matmul(@sess, blk.t_w_gate, t_h2)        # ne=[d_ff, 1]
    t_up     = TinyNNCuda.tnn_matmul(@sess, blk.t_w_up,   t_h2)        # ne=[d_ff, 1]
    t_silug  = TinyNNCuda.tnn_silu(@sess, t_gate)
    t_gated  = TinyNNCuda.tnn_mul(@sess, t_silug, t_up)
    t_dn     = TinyNNCuda.tnn_matmul(@sess, blk.t_w_down, t_gated)     # ne=[d_model, 1]

    TinyNNCuda.tnn_add(@sess, t_x_attn, t_dn)
  end

  # One query head. Uses the (already-written) K and V of the
  # corresponding KV head — index = hq / group_size.
  def build_attention_qhead_step(t_h, blk, hq, t_pos, pos, scale,
                                  bytes_d_head, bytes_max_T)
    hkv = hq / @group_size

    t_q_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_q[hq], t_h)   # ne=[d_head, 1]
    if @has_qkv_bias
      t_q_pre = TinyNNCuda.tnn_add(@sess, t_q_raw, blk.t_b_q[hq])
    else
      t_q_pre = t_q_raw
    end
    t_q     = TinyNNCuda.tnn_rope_ext(@sess, t_q_pre, t_pos, @d_head, @rope_base)

    t_K_hist = TinyNNCuda.tnn_view_2d(@sess, blk.t_K[hkv],
                                    @d_head, pos + 1, bytes_d_head, 0)
    t_V_hist = TinyNNCuda.tnn_view_2d(@sess, blk.t_V[hkv],
                                    pos + 1, @d_head, bytes_max_T, 0)

    t_scores = TinyNNCuda.tnn_matmul(@sess, t_K_hist, t_q)
    t_scaled = TinyNNCuda.tnn_scale(@sess, t_scores, scale)
    t_attn   = TinyNNCuda.tnn_softmax(@sess, t_scaled)
    TinyNNCuda.tnn_matmul(@sess, t_V_hist, t_attn)
  end
end

# Init-param names deliberately differ from the ivar names — same
# defensive pattern as GPT2KVStepResult.
class SmolLM2KVStepResultCuda
  attr_accessor :t_token_id, :t_pos, :kv_step_logits
  def initialize(tok_ptr, pos_ptr, logits_ptr)
    @t_token_id     = tok_ptr
    @t_pos          = pos_ptr
    @kv_step_logits = logits_ptr
  end
end

module SmolLM2KVCuda
  # Upload all Toy::SmolLM2 weights into a realized cache (+ zero-init
  # the K/V buffers).
  def self.upload_from(kv_cache, model)
    sess     = kv_cache.sess
    n        = kv_cache.n_layers
    n_heads  = kv_cache.n_heads
    n_kv     = kv_cache.n_kv
    d_model  = kv_cache.d_model
    d_head   = kv_cache.d_head
    max_T    = kv_cache.max_T

    TinyNNCuda.upload_row_major(sess, kv_cache.t_token_embed, model.token_embed.weight)
    TinyNNCuda.tnn_upload_from_float_array(sess, kv_cache.t_final_norm_gamma,
                                        model.final_norm.gamma, d_model)
    if kv_cache.has_untied_output
      TinyNNCuda.upload_row_major(sess, kv_cache.t_output, model.output_proj)
    end

    kv_zero_k = Mat.new(max_T,  d_head)
    kv_zero_v = Mat.new(d_head, max_T)

    li = 0
    while li < n
      blk_n = model.stack[li]
      blk_f = kv_cache.kv_blocks_ffi[li]

      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_rn1_gamma, blk_n.rn1.gamma, d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_rn2_gamma, blk_n.rn2.gamma, d_model)

      hq = 0
      while hq < n_heads
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_q[hq], blk_n.attn.w_q[hq])
        if kv_cache.has_qkv_bias
          TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_q[hq], blk_n.attn.b_q[hq], d_head)
        end
        hq = hq + 1
      end

      hkv = 0
      while hkv < n_kv
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_k[hkv], blk_n.attn.w_k[hkv])
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_v[hkv], blk_n.attn.w_v[hkv])
        if kv_cache.has_qkv_bias
          TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_k[hkv], blk_n.attn.b_k[hkv], d_head)
          TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_v[hkv], blk_n.attn.b_v[hkv], d_head)
        end
        TinyNNCuda.upload_row_major(sess, blk_f.t_K[hkv], kv_zero_k)
        TinyNNCuda.upload_row_major(sess, blk_f.t_V[hkv], kv_zero_v)
        hkv = hkv + 1
      end

      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_o,    blk_n.attn.w_o)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_gate, blk_n.ffn.w_gate)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_up,   blk_n.ffn.w_up)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_down, blk_n.ffn.w_down)

      li = li + 1
    end
  end

  # Decode one new token at position `pos`. Returns the (1, vocab)
  # logits Mat for the new position.
  def self.decode_step(kv_cache, token_id, pos)
    TinyNNCuda.tnn_reset_for_rebuild(kv_cache.sess)
    step = kv_cache.build_decode_step(pos)
    TinyNNCuda.tnn_realize(kv_cache.sess, step.kv_step_logits)
    TinyNNCuda.upload_int_array(kv_cache.sess, step.t_token_id, [token_id])
    TinyNNCuda.upload_int_array(kv_cache.sess, step.t_pos,      [pos])
    TinyNNCuda.tnn_compute(kv_cache.sess)
    TinyNNCuda.download_row_major(kv_cache.sess, step.kv_step_logits, 1, kv_cache.vocab_size)
  end
end
