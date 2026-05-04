# Prep step (CRuby): download a HuggingFace dataset, tokenize it with the
# French/English tokenizer, and write three flat-text files that the
# Spinel-compiled training program can read without any HTTP / Unicode /
# regex machinery of its own:
#
#   data/ts_vocab.txt    — one word per line; line index = token id
#   data/ts_seqs.txt     — one sequence per line, space-separated token IDs
#   data/ts_prompt.txt   — the seed prompt's token IDs, space-separated
#
# Usage:
#   ruby prep_tinystories.rb \
#     --repo roneneldan/TinyStories \
#     --file TinyStoriesV2-GPT4-valid.txt \
#     --max_lines 500 \
#     --context_length 64 \
#     --prompt "Once upon a time"
require "optparse"
require "fileutils"
require_relative "dataset_loader"
require_relative "tokenizer"

opts = {
  repo:           "roneneldan/TinyStories",
  file:           "TinyStoriesV2-GPT4-valid.txt",
  max_lines:      500,
  context_length: 64,
  prompt:         "Once upon a time",
  out_dir:        "data",
}
OptionParser.new do |o|
  o.on("--repo R")               { |v| opts[:repo] = v }
  o.on("--file F")               { |v| opts[:file] = v }
  o.on("--max_lines N", Integer) { |v| opts[:max_lines] = v }
  o.on("--context_length N", Integer) { |v| opts[:context_length] = v }
  o.on("--prompt P")             { |v| opts[:prompt] = v }
  o.on("--out_dir D")            { |v| opts[:out_dir] = v }
end.parse!

FileUtils.mkdir_p(opts[:out_dir])

puts "Loading #{opts[:repo]} / #{opts[:file]} (first #{opts[:max_lines]} lines)…"
lines = DatasetLoader.head(opts[:repo], opts[:file], opts[:max_lines])
puts "  #{lines.size} non-empty lines"

# Build a single shared vocabulary from the corpus + the prompt so any
# prompt token has an ID in the trained model.
tokenized      = lines.map { |l| Tokenizer.tokenize_french(l) }
prompt_tokens  = Tokenizer.tokenize_french(opts[:prompt])
vocab_words    = (tokenized.flatten + prompt_tokens).uniq
word_to_index  = vocab_words.each_with_index.to_h
puts "  vocab size: #{vocab_words.size}"

# Tokenize each line into IDs, then chunk over-long lines into windows
# of at most context_length tokens (same trick we use in CRuby).
sequences = tokenized.flat_map do |toks|
  ids = toks.map { |w| word_to_index[w] }.compact
  ids.each_slice(opts[:context_length]).to_a
end.reject { |s| s.size < 2 }
puts "  sequences (after chunking): #{sequences.size}"

# Write the three files.
File.write(
  File.join(opts[:out_dir], "ts_vocab.txt"),
  vocab_words.join("\n") + "\n"
)
File.open(File.join(opts[:out_dir], "ts_seqs.txt"), "w") do |f|
  sequences.each { |seq| f.puts seq.join(" ") }
end
prompt_ids = prompt_tokens.map { |w| word_to_index[w] }.compact
File.write(
  File.join(opts[:out_dir], "ts_prompt.txt"),
  prompt_ids.join(" ") + "\n"
)

puts "Wrote #{opts[:out_dir]}/ts_vocab.txt, ts_seqs.txt, ts_prompt.txt"
puts "Prompt token IDs: #{prompt_ids.inspect}"
