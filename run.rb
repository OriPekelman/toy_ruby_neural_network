require "optparse"
require_relative "neural_network"

options = {
  epochs: 10,
  learning_rate: 0.01,
  corpus: "simple_french",
  prompt: "Je",
  hidden_size: 4,
  latent_size: 2,
  num_tokens: 4,
}

parser = OptionParser.new
parser.on("--epochs EPOCHS", Integer) { |o| options[:epochs] = o }
parser.on("--learning_rate LEARNING_RATE", Float) { |o| options[:learning_rate] = o }
parser.on("--corpus CORPUS") { |o| options[:corpus] = o }
parser.on("--prompt PROMPT") { |o| options[:prompt] = o }
parser.on("--hidden_size HIDDEN_SIZE", Integer) { |o| options[:hidden_size] = o }
parser.on("--latent_size LATENT_SIZE", Integer) { |o| options[:latent_size] = o }
parser.on("--num_tokens TOKENS", Integer) { |o| options[:num_tokens] = o }
parser.parse!(into: options)

nn_file_name = "#{options[:corpus]}_#{options[:hidden_size]}_#{options[:latent_size]}_#{options[:epochs]}.dat"

# Train or load the network
if File.file?(nn_file_name)
  puts "Loading existing model"
  nn = NeuralNetwork.load_from_file(nn_file_name)
else
  puts "Training new model"

  # Read training data
  data = File.readlines("#{options[:corpus]}.txt", chomp: true)

  nn, vectors = NeuralNetwork.create_from_data(data, options[:hidden_size], options[:latent_size])

  nn.train(vectors, options[:epochs], options[:learning_rate])
  nn.save_to_file(nn_file_name)
end

puts "\nCompletion for prompt: '#{options[:prompt]}'"
# the other option :deterministic is not very interesting.
puts nn.generate_completion(options[:prompt], options[:num_tokens], :probabilistic)