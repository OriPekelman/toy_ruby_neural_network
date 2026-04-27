require_relative "transformer"

# Standalone Spinel-compilable entry point: train the transformer on the
# minimal corpus and generate a continuation. No CLI, no flags — keep it
# simple so the AOT compiler's job is small.

CORPUS = [
  "un est deux",
  "deux est trois",
  "trois est quatre",
  "quatre est cinq",
  "cinq est six",
]

srand(42)

nn, sequences = TransformerLM.create_from_data(
  CORPUS,
  d_model:        16,
  d_ff:           32,
  n_heads:        2,
  context_length: 8,
  n_layers:       2,
)

nn.train(sequences, epochs: 30, learning_rate: 0.005, batch_size: 32)

puts ""
puts "Generation from 'un est':"
puts nn.generate("un est", max_tokens: 4, temperature: 0.7)
