require "optparse"
require "benchmark"
require_relative "transformer"
require_relative "neural_network"

def print_size_info(nn)
  info = nn.model_size
  puts "\nModel: #{info[:parameters]} parameters (#{info[:memory_kb]} KB)"
  info[:layers].each { |k, v| puts "  #{k}: #{v}" }
end

options = {
  model:          "transformer",
  epochs:         20,
  learning_rate:  0.005,    # good default for Adam; autoencoder uses 0.05
  corpus:         "minimal",
  prompt:         "un",
  num_tokens:     4,
  batch_size:     32,
  temperature:    0.8,

  # transformer hyperparams
  d_model:        16,
  d_ff:           32,
  n_heads:        2,
  n_layers:       2,
  context_length: 16,

  # autoencoder hyperparams
  hidden_size:    8,
  latent_size:    4,
}

OptionParser.new do |o|
  o.on("--model MODEL", %w[transformer autoencoder]) { |v| options[:model] = v }
  o.on("--epochs N",         Integer) { |v| options[:epochs] = v }
  o.on("--learning_rate F",  Float)   { |v| options[:learning_rate] = v }
  o.on("--corpus NAME")               { |v| options[:corpus] = v }
  o.on("--prompt P")                  { |v| options[:prompt] = v }
  o.on("--num_tokens N",     Integer) { |v| options[:num_tokens] = v }
  o.on("--batch_size N",     Integer) { |v| options[:batch_size] = v }
  o.on("--temperature F",    Float)   { |v| options[:temperature] = v }

  o.on("--d_model N",        Integer) { |v| options[:d_model] = v }
  o.on("--d_ff N",           Integer) { |v| options[:d_ff] = v }
  o.on("--n_heads N",        Integer) { |v| options[:n_heads] = v }
  o.on("--n_layers N",       Integer) { |v| options[:n_layers] = v }
  o.on("--context_length N", Integer) { |v| options[:context_length] = v }

  o.on("--hidden_size N",    Integer) { |v| options[:hidden_size] = v }
  o.on("--latent_size N",    Integer) { |v| options[:latent_size] = v }
end.parse!(into: options)

corpus_path = "#{options[:corpus]}.txt"
data        = File.readlines(corpus_path, chomp: true).reject { |l| l.strip.empty? }

case options[:model]
when "transformer"
  cache_file = "#{options[:corpus]}_tx_dm#{options[:d_model]}_dff#{options[:d_ff]}_" \
               "h#{options[:n_heads]}_L#{options[:n_layers]}_ctx#{options[:context_length]}_e#{options[:epochs]}.txt"

  if File.file?(cache_file)
    nn = TransformerLM.load_from_file(cache_file)
  else
    nn, sequences = TransformerLM.create_from_data(
      data,
      d_model:        options[:d_model],
      d_ff:           options[:d_ff],
      n_heads:        options[:n_heads],
      context_length: options[:context_length],
      n_layers:       options[:n_layers],
    )
    nn.train(sequences,
             epochs:        options[:epochs],
             learning_rate: options[:learning_rate],
             batch_size:    options[:batch_size])
    nn.save_to_file(cache_file)
  end

  print_size_info(nn)

  puts "\nGenerating from prompt: '#{options[:prompt]}'"
  Benchmark.measure {
    puts nn.generate(options[:prompt],
                     max_tokens:  options[:num_tokens],
                     temperature: options[:temperature])
  }.tap { |t| puts "Generation time: #{t.real.round(4)}s" }

when "autoencoder"
  cache_file = "#{options[:corpus]}_ae_h#{options[:hidden_size]}_l#{options[:latent_size]}_e#{options[:epochs]}.dat"

  if File.file?(cache_file)
    nn = NeuralNetwork.load_from_file(cache_file)
  else
    nn, vectors = NeuralNetwork.create_from_data(data, options[:hidden_size], options[:latent_size])
    nn.train(vectors, options[:epochs], options[:learning_rate], batch_size: options[:batch_size])
    nn.save_to_file(cache_file)
  end

  print_size_info(nn)

  puts "\nCompletion for prompt: '#{options[:prompt]}'"
  Benchmark.measure {
    puts nn.generate_completion(options[:prompt], options[:num_tokens], :probabilistic)
  }.tap { |t| puts "Generation time: #{t.real.round(4)}s" }
end

