# Spinel-compiled TinyStories training run.
#
# Reads the three files written by prep_tinystories.rb:
#   data/ts_vocab.txt    — words, one per line
#   data/ts_seqs.txt     — token-id sequences, one per line, space-separated
#   data/ts_prompt.txt   — seed prompt's token IDs, single line
#
# Trains a small transformer end-to-end and prints a continuation of the
# prompt. Hardcoded hyperparameters (edit and recompile to retune):
#   d_model=32, d_ff=64, n_heads=4, n_layers=2, ctx=64
#   epochs=30, batch_size=32, lr=0.005, temperature=0.8
require_relative "transformer_spinel"

# Hyperparameters — keep in lock-step with prep_tinystories.rb's
# --context_length flag.
D_MODEL        = 16
D_FF           = 32
N_HEADS        = 2
N_LAYERS       = 2
CONTEXT_LENGTH = 64
EPOCHS         = 25
BATCH_SIZE     = 32
LR             = 0.005
NUM_TOKENS     = 30
TEMPERATURE    = 0.7

# ---- Load vocabulary ------------------------------------------------------
# Spinel doesn't support File.readlines or chained enumerators like
# `each_line.first`. We use File.open + each_line, with a literal-seed
# array that pop'd; pop works for top-level StrArray.

vocab = ["?"]
vocab.pop
File.open("data/ts_vocab.txt", "r") do |f|
  f.each_line do |line|
    vocab.push(line.chomp)
  end
end
puts "vocab: " + vocab.length.to_s + " words"

# Helper: parse "n n n" → IntArray of token IDs.
def parse_ids(line)
  parts = line.split(" ")
  ids   = [parts[0].to_i]
  k = 1
  while k < parts.length
    ids.push(parts[k].to_i)
    k += 1
  end
  ids
end

# ---- Load sequences ------------------------------------------------------
# `pop` on an Array-of-IntArray is a no-op in Spinel, so we can't seed and
# pop the way we did for vocab. Instead read all raw lines into a typed
# StrArray first, then parse each line's IDs and push the resulting
# IntArray onto a sequences container that's seeded from the FIRST line.

seq_lines = ["?"]
seq_lines.pop
File.open("data/ts_seqs.txt", "r") do |f|
  f.each_line { |line| seq_lines.push(line.chomp) }
end

sequences = [parse_ids(seq_lines[0])]
si = 1
while si < seq_lines.length
  sequences.push(parse_ids(seq_lines[si]))
  si += 1
end
puts "sequences: " + sequences.length.to_s

# ---- Load prompt ----------------------------------------------------------

prompt_lines = ["?"]
prompt_lines.pop
File.open("data/ts_prompt.txt", "r") do |f|
  f.each_line { |line| prompt_lines.push(line.chomp) }
end
prompt_ids = parse_ids(prompt_lines[0])
puts "prompt token IDs: " + prompt_ids.length.to_s

# ---- Build model ----------------------------------------------------------

model = TransformerLM.new(vocab.length, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT_LENGTH)
model.vocabulary = vocab
puts "Built model: " + model.vocab_size.to_s + " vocab, " +
                       D_MODEL.to_s + " d_model, " +
                       N_LAYERS.to_s + " layers, ctx " + CONTEXT_LENGTH.to_s

# ---- Training loop --------------------------------------------------------

grads = Gradients.new(vocab.length, D_MODEL, D_FF, N_HEADS, D_MODEL / N_HEADS, N_LAYERS, CONTEXT_LENGTH)

# Warm-up: anchor class-method param types via top-level calls.
# Spinel only picks up class-method parameter types from top-level call
# sites; inside class methods, calls to other class methods inherit
# whatever default Spinel guessed (mrb_int for unfilled). One forward +
# one backward against the prompt is enough to fix the chain.
puts "Warm-up forward + backward to anchor param inference…"
model.forward(prompt_ids)
model.backward(prompt_ids, grads)

puts ""
puts "Training " + EPOCHS.to_s + " epochs over " + sequences.length.to_s + " sequences (lr=" + LR.to_s + ", batch=" + BATCH_SIZE.to_s + ")"

t_start  = Time.now
epoch    = 0
while epoch < EPOCHS
  total_loss = 0.0
  n_steps    = 0

  # Mini-batch by walking through sequences and zeroing grads at each
  # batch boundary. (We don't shuffle here — for a fixed-time benchmark
  # epoch ordering doesn't matter much; shuffling under Spinel would
  # need an Array#shuffle that the gem currently doesn't expose.)
  bi = 0
  while bi < sequences.length
    grads.fill_zero
    batch_end = bi + BATCH_SIZE
    if batch_end > sequences.length
      batch_end = sequences.length
    end
    inner = bi
    while inner < batch_end
      seq = sequences[inner]
      sl  = seq.length              # type hint: seq is an IntArray
      if sl >= 2 && sl <= CONTEXT_LENGTH
        # Inline train_step's body. Going through `model.train_step(seq, ...)`
        # gets seq inferred as int by Spinel for reasons we haven't pinned
        # down — calling forward/backward directly here keeps the IntArray
        # type stable across the call chain.
        grads.fill_zero
        model.forward(seq)
        model.backward(seq, grads)
        model.apply_gradients_sgd(grads, LR)
        total_loss = total_loss + grads.loss
        n_steps    = n_steps    + 1
      end
      inner += 1
    end
    bi = batch_end
  end

  if n_steps > 0
    mean = total_loss / n_steps
    puts "epoch " + (epoch + 1).to_s + "/" + EPOCHS.to_s + "  mean_loss=" + mean.to_s
  end
  epoch += 1
end
elapsed = Time.now - t_start
puts ""
puts "Training completed in " + elapsed.to_s + "s"

# ---- Generation -----------------------------------------------------------

puts ""
puts "Generating " + NUM_TOKENS.to_s + " tokens at temperature " + TEMPERATURE.to_s + "…"
all_tokens = model.generate_from_ids(prompt_ids, NUM_TOKENS, TEMPERATURE)

# Print as words separated by spaces.
out = vocab[all_tokens[0]]
i = 1
while i < all_tokens.length
  out = out + " " + vocab[all_tokens[i]]
  i += 1
end
puts out
