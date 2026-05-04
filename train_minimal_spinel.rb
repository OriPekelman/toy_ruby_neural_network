# Original Spinel-compat smoke test: build a 4 K-parameter transformer,
# run forward + backward, take 40 SGD steps over 3 hardcoded sequences.
# Expected loss curve: ~2.0 → ~0.03 in well under a second when compiled.
require_relative "transformer_spinel"

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

loss_res = model.cross_entropy_grad(logits, token_ids)
puts "  loss="
puts loss_res.loss
puts "  dlogits shape="   + loss_res.dlogits.nrows.to_s + "x" + loss_res.dlogits.ncols.to_s
puts "  dlogits.flat[0]="
puts loss_res.dlogits.flat[0]
puts "  dlogits.flat[6]="
puts loss_res.dlogits.flat[6]

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

  grads.fill_zero
  model.forward(seqs_a)
  model.backward(seqs_a, grads)
  model.apply_gradients_sgd(grads, 0.05)
  total_loss = total_loss + grads.loss

  grads.fill_zero
  model.forward(seqs_b)
  model.backward(seqs_b, grads)
  model.apply_gradients_sgd(grads, 0.05)
  total_loss = total_loss + grads.loss

  grads.fill_zero
  model.forward(seqs_c)
  model.backward(seqs_c, grads)
  model.apply_gradients_sgd(grads, 0.05)
  total_loss = total_loss + grads.loss

  if step % 4 == 0
    puts "  step="
    puts step
    puts "  mean_loss="
    puts total_loss / 3.0
  end
  step += 1
end
