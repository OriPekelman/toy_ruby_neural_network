# Spinel-compiled TinyStories language-model training.
#
# Pipeline:
#   prep/prep_tinystories.rb   (CRuby) writes data/ts_*.txt
#   this script                (Spinel) trains and prints a continuation
#
# Architecture:
#   pre-RMSNorm transformer · multi-head causal attention · GeLU FFN
#   tied input/output embeddings · Adam · linear warmup → cosine decay
require_relative "../lib/transformer"
require_relative "../lib/training"

# ---- Hyperparameters ------------------------------------------------------

D_MODEL        = 32
D_FF           = 64
N_HEADS        = 4
N_LAYERS       = 2
CONTEXT_LENGTH = 64       # match prep_tinystories.rb's --context_length
BATCH_SIZE     = 32
EPOCHS         = 60

MAX_LR         = 0.001
LR_MIN         = 0.00001
WARMUP_STEPS   = 200
ADAM_BETA1     = 0.9
ADAM_BETA2     = 0.999
ADAM_EPS       = 0.00000001

NUM_TOKENS     = 60
TEMPERATURE    = 0.7

# ---- Load tokenized corpus ------------------------------------------------

vocab      = read_vocab("data/ts_vocab.txt")
sequences  = read_sequences("data/ts_seqs.txt")
prompt_ids = read_prompt("data/ts_prompt.txt")
puts "Loaded vocab=" + vocab.length.to_s +
     ", sequences=" + sequences.length.to_s +
     ", prompt_tokens=" + prompt_ids.length.to_s

# ---- Build model + grads + optimizer + data loader + LR schedule ---------

model     = TransformerLM.new(vocab.length, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT_LENGTH)
model.vocabulary = vocab

grads     = Gradients.new(vocab.length, D_MODEL, D_FF, N_HEADS, D_MODEL / N_HEADS, N_LAYERS, CONTEXT_LENGTH)
optimizer = Adam.new(model, ADAM_BETA1, ADAM_BETA2, ADAM_EPS)
loader    = DataLoader.new(sequences, BATCH_SIZE)
schedule  = LRSchedule.new(WARMUP_STEPS, EPOCHS * sequences.length, MAX_LR, LR_MIN)

# Spinel anchors class-method param types from top-level call sites.
# Doing one full forward + backward + adam against the prompt fixes the
# whole call chain end-to-end before training starts. The reset clears
# the warm-up step's contribution out of the optimizer state.
puts "Warm-up forward + backward + adam to anchor param inference…"
model.forward(prompt_ids)
model.backward(prompt_ids, grads)
optimizer.step(grads, MAX_LR)
optimizer.reset

# ---- Train ----------------------------------------------------------------

puts ""
puts "Training " + EPOCHS.to_s + " epochs over " + sequences.length.to_s +
     " sequences (max_lr=" + MAX_LR.to_s +
     ", warmup=" + WARMUP_STEPS.to_s +
     ", batch=" + BATCH_SIZE.to_s + ")"

t_start = Time.now
step    = 0
epoch   = 0
while epoch < EPOCHS
  total_loss = 0.0
  n          = 0

  bi = 0
  while bi < loader.batch_count
    grads.fill_zero
    s_start = loader.batch_start(bi)
    s_end   = loader.batch_end(bi)

    inner = s_start
    while inner < s_end
      # Read directly from `sequences`; Spinel infers PtrArray<IntArray>
      # cleanly here, but routing through `loader.at(inner)` confuses
      # the class-method return-type inference and Spinel guesses Float.
      seq = sequences[inner]
      sl  = seq.length
      if sl >= 2 && sl <= CONTEXT_LENGTH
        grads.fill_zero
        model.forward(seq)
        model.backward(seq, grads)
        optimizer.step(grads, schedule.at(step))
        total_loss += grads.loss
        n          += 1
        step       += 1
      end
      inner += 1
    end
    bi += 1
  end

  if n > 0
    puts "epoch " + (epoch + 1).to_s + "/" + EPOCHS.to_s +
         "  mean_loss=" + (total_loss / n).to_s +
         "  lr=" + schedule.at(step).to_s
  end
  epoch += 1
end

elapsed = Time.now - t_start
puts ""
puts "Training completed in " + elapsed.to_s + "s"

# ---- Generate -------------------------------------------------------------

puts ""
puts "Generating " + NUM_TOKENS.to_s + " tokens at temperature " + TEMPERATURE.to_s + "…"
ids = model.generate_from_ids(prompt_ids, NUM_TOKENS, TEMPERATURE)

out = vocab[ids[0]]
i   = 1
while i < ids.length
  out = out + " " + vocab[ids[i]]
  i  += 1
end
puts out
