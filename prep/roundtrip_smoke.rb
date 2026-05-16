#!/usr/bin/env ruby
# frozen_string_literal: true
# prep/roundtrip_smoke.rb — verify: model → algorithm_card → parser → model.
#
# We can't `eval` the parser output in CRuby (Toy::* requires the
# Spinel ffi_lib loader). Instead this smoke:
#   1. Reads our two known canonical cards from data/_cards/
#      (generated once via `./demos/algorithm_cards > data/_cards/all.txt`)
#   2. Parses each through prep/card_to_code.rb (as a library)
#   3. Extracts the hyperparameters and compares them to the
#      expected values for each model family.
#
# Pass = parser correctly round-trips our own emissions back to
# constructor arguments.

require_relative "card_to_code"

EXPECTED = {
  smollm2: {
    "V"      => 49152,  "D"  => 576,  "H" => 9,    "H_kv" => 3,
    "D_f"    => 1536,   "N"  => 30,   "ctx" => 8192,
    "θ_base" => 100000.0,
  },
  gpt2: {
    "V"   => 50257, "D"   => 768, "H" => 12,
    "D_f" => 3072,  "N"   => 6,   "ctx" => 1024,
  },
}

def check(card_text, family_expected)
  family = detect_family(card_text) or
    fail "could not detect family in card"
  hyper = parse_hyper(card_text)
  raise "family mismatch: got #{family}, expected #{family_expected}" if family != family_expected
  expected = EXPECTED.fetch(family)
  expected.each do |k, v|
    actual = hyper[k]
    raise "[#{family}] key #{k}: expected #{v.inspect}, got #{actual.inspect}" if actual != v
  end
  puts "✓ #{family}: #{expected.size} hyperparameters round-tripped"
end

cards_path = ARGV[0] || "data/_cards/all.txt"
abort "missing #{cards_path} — run `./demos/algorithm_cards > #{cards_path}` first" unless File.exist?(cards_path)

text = File.read(cards_path)

# Split on "Algorithm: Toy::" boundaries (cards are concatenated in
# demos/algorithm_cards output).
chunks = text.split(/(?=Algorithm: Toy::(?:GPT2|SmolLM2)\.forward)/)
chunks.shift if chunks.first && chunks.first !~ /Algorithm: Toy::/

raise "expected 2 top-level cards, got #{chunks.size}" if chunks.size != 2

chunks.each do |chunk|
  family = chunk.include?("HF GPT-2 family") ? :gpt2 : :smollm2
  check(chunk, family)
end

puts "round-trip smoke: ok"
