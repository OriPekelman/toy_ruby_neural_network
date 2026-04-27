require "matrix"

class NeuralNetwork
  # An autoencoder with one bottleneck:
  #
  #   input (V) ──W1──▶ h1 (H) ──W2──▶ z (L) ──W3──▶ h2 (H) ──W4──▶ output (V)
  #
  # All weight matrices are stored "fan-in × fan-out" so that the forward pass
  # is `next = act(W.transpose * prev)`. ReLU on the hidden/latent layers,
  # sigmoid on the output. Loss is 0.5 * sum((target - output)^2).
  #
  # Training is a clean three-step cycle per example:
  #   1. forward  — compute and cache every layer's activations
  #   2. backward — use those cached activations to compute one gradient
  #                 per weight matrix (returned as a hash, NOT applied)
  #   3. apply_gradients — actually write the update onto the weights
  #
  # Mini-batch training is just: zero an accumulator, add each example's
  # gradients into it, divide by batch size, apply once. That accumulation
  # step is the whole point of mini-batch / batch GD — averaging the
  # *gradients*, never the raw errors.

  attr_reader :vocabulary, :word_to_index, :hidden_layer_1, :hidden_layer_2, :latent_representation

  def initialize(input_size, hidden_size, latent_size, vocabulary = nil, word_to_index = nil)
    @input_size = input_size
    @hidden_size = hidden_size
    @latent_size = latent_size
    @vocabulary = vocabulary
    @word_to_index = word_to_index

    # Random init in [-1, 1]: breaks symmetry so neurons learn different
    # features, and keeps activations in a range where the gradient is alive.
    @weights_input_to_hidden  = Matrix.build(input_size,  hidden_size) { rand(-1.0..1.0) }
    @weights_hidden_to_latent = Matrix.build(hidden_size, latent_size) { rand(-1.0..1.0) }
    @weights_latent_to_hidden = Matrix.build(latent_size, hidden_size) { rand(-1.0..1.0) }
    @weights_hidden_to_output = Matrix.build(hidden_size, input_size)  { rand(-1.0..1.0) }
  end

  def self.create_from_data(data, hidden_size, latent_size)
    temp_nn = new(1, 1, 1)
    words = data.flat_map { |line| temp_nn.tokenize_french(line) }.uniq
    word_to_index = words.each_with_index.to_h
    nn = new(words.size, hidden_size, latent_size, words, word_to_index)
    [nn, nn.text_to_vectors(data)]
  end

  def text_to_vectors(data)
    data.map { |text| text_to_vector(text) }
  end

  def tokenize_french(text)
    text = text.unicode_normalize(:nfkc).downcase
    tokens = text.split(/\b/)

    processed_tokens = []
    tokens.each_with_index do |token, i|
      next if token.nil? || token.strip.empty?

      if token == "'" && i > 0 && i < tokens.length - 1
        processed_tokens[-1] += token + tokens[i + 1].to_s
        tokens[i + 1] = nil
      elsif token == "-" && i > 0 && i < tokens.length - 1
        processed_tokens[-1] += token + tokens[i + 1].to_s
        tokens[i + 1] = nil
      elsif !token.match?(/\A[[:punct:]]+\z/)
        processed_tokens << token
      end
    end

    processed_tokens.map! { |token| token.gsub(/\A[[:punct:]]+|[[:punct:]]+\z/, "") }
    processed_tokens.reject! { |token| token.strip.empty? }
    processed_tokens
  end

  def text_to_vector(text)
    vector = Array.new(@vocabulary.size, 0)
    tokenize_french(text).each do |word|
      index = @word_to_index[word]
      vector[index] = 1 if index
    end
    vector
  end

  # ------------------------------------------------------------------
  #  Forward pass — caches every activation that backward will need.
  # ------------------------------------------------------------------
  def forward(input)
    @input_vector = Matrix.column_vector(input)

    @hidden_layer_1        = relu(@weights_input_to_hidden.transpose  * @input_vector)
    @latent_representation = relu(@weights_hidden_to_latent.transpose * @hidden_layer_1)
    @hidden_layer_2        = relu(@weights_latent_to_hidden.transpose * @latent_representation)
    @output                = sigmoid(@weights_hidden_to_output.transpose * @hidden_layer_2)

    [@output, @latent_representation]
  end

  # ------------------------------------------------------------------
  #  Backward pass — pure function of the cached activations.
  #
  #  For each layer we compute `delta`, the gradient of the loss with
  #  respect to that layer's *pre-activation*. Then the gradient for the
  #  weights feeding into that layer is `prev_activation * delta.transpose`.
  #
  #  Sign convention: we use `error = target - output` and then ADD the
  #  update (gradient ascent on -loss == gradient descent on loss).
  #
  #  Chain rule, layer by layer:
  #    delta_out = (target - output) * sigmoid'(z_out)
  #              = error * output * (1 - output)        # note: post-activation, NOT error
  #    delta_h2  = (W4 * delta_out) ⊙ relu'(h2)
  #    delta_z   = (W3 * delta_h2)  ⊙ relu'(z)
  #    delta_h1  = (W2 * delta_z)   ⊙ relu'(h1)
  # ------------------------------------------------------------------
  def backward(target)
    error = Matrix.column_vector(target) - @output

    delta_output = error.map.with_index do |e, i|
      e * @output[i, 0] * (1 - @output[i, 0])
    end

    delta_hidden_2 = (@weights_hidden_to_output * delta_output).map.with_index do |e, i|
      e * relu_derivative(@hidden_layer_2[i, 0])
    end

    delta_latent = (@weights_latent_to_hidden * delta_hidden_2).map.with_index do |e, i|
      e * relu_derivative(@latent_representation[i, 0])
    end

    delta_hidden_1 = (@weights_hidden_to_latent * delta_latent).map.with_index do |e, i|
      e * relu_derivative(@hidden_layer_1[i, 0])
    end

    {
      weights_input_to_hidden:  @input_vector          * delta_hidden_1.transpose,
      weights_hidden_to_latent: @hidden_layer_1        * delta_latent.transpose,
      weights_latent_to_hidden: @latent_representation * delta_hidden_2.transpose,
      weights_hidden_to_output: @hidden_layer_2        * delta_output.transpose,
      loss: 0.5 * error.map { |e| e * e }.to_a.flatten.sum,
    }
  end

  # Apply an already-computed gradient update. Used by both SGD (one
  # example) and mini-batch (averaged across the batch).
  def apply_gradients(gradients, learning_rate)
    @weights_input_to_hidden  += learning_rate * gradients[:weights_input_to_hidden]
    @weights_hidden_to_latent += learning_rate * gradients[:weights_hidden_to_latent]
    @weights_latent_to_hidden += learning_rate * gradients[:weights_latent_to_hidden]
    @weights_hidden_to_output += learning_rate * gradients[:weights_hidden_to_output]
  end

  # ------------------------------------------------------------------
  #  Training loop
  # ------------------------------------------------------------------
  def train(inputs, epochs, learning_rate, batch_size: 32, method: :mini_batch)
    puts "Training using #{method.to_s.gsub("_", " ").capitalize}"
    puts "Epochs: #{epochs}, Learning rate: #{learning_rate}, Batch size: #{batch_size}"
    total_start_time = Time.now

    epochs.times do |epoch|
      epoch_loss =
        case method
        when :sgd        then run_sgd(inputs, learning_rate)
        when :batch      then run_mini_batch(inputs, learning_rate, inputs.size)
        when :mini_batch then run_mini_batch(inputs, learning_rate, batch_size)
        end
      puts "Epoch #{epoch + 1}/#{epochs}  loss=#{(epoch_loss / inputs.size).round(6)}"
    end

    puts "\nTraining completed in #{(Time.now - total_start_time).round(2)} seconds"
  end

  def encode(input)
    input_vector = Matrix.column_vector(input)
    hidden_layer = relu(@weights_input_to_hidden.transpose * input_vector)
    relu(@weights_hidden_to_latent.transpose * hidden_layer).to_a.flatten
  end

  def decode(latent_vector)
    latent_matrix = Matrix.column_vector(latent_vector.to_a.flatten)
    hidden_layer = relu(@weights_latent_to_hidden.transpose * latent_matrix)
    sigmoid(@weights_hidden_to_output.transpose * hidden_layer).to_a.flatten
  end

  def vector_to_words(vector, threshold = 0.5)
    raise "Vocabulary not initialized" if @vocabulary.nil?
    vector.each_with_index.map { |v, i| v > threshold ? @vocabulary[i] : nil }.compact.join(" ")
  end

  def vector_to_words_probabilistic(vector, threshold = 0.1)
    raise "Vocabulary not initialized" if @vocabulary.nil?
    vector.each_with_index.map { |v, i| rand < v + threshold ? @vocabulary[i] : nil }.compact.join(" ")
  end

  def vector_to_single_word(vector, threshold = 0.5)
    raise "Vocabulary not initialized" if @vocabulary.nil?
    index = vector.index { |v| v > threshold }
    index ? @vocabulary[index] : nil
  end

  def vector_to_single_word_probabilistic(vector, threshold = 0.1)
    raise "Vocabulary not initialized" if @vocabulary.nil?
    probabilities = vector.map { |v| v + threshold }
    chosen_index = probabilities.each_with_index.max_by { |prob, _| rand ** (1.0 / prob) }.last
    @vocabulary[chosen_index]
  end

  def generate_phrase(latent_vector, method = :deterministic)
    generated_input = decode(latent_vector)
    case method
    when :deterministic then vector_to_words(generated_input)
    when :probabilistic then vector_to_words_probabilistic(generated_input)
    else raise "Unknown method: #{method}"
    end
  end

  def generate_completion(prompt, max_tokens = 20, method = :deterministic)
    prompt_vector = text_to_vector(prompt)
    encoded_prompt = encode(prompt_vector).to_a.flatten

    completion = prompt.split

    max_tokens.times do
      adjusted_latent = encoded_prompt.map { |v| v + rand(-0.1..0.1) }
      next_word = generate_next_token(adjusted_latent, method)
      break if next_word.nil? || next_word.empty?

      completion << next_word
      new_token_vector = text_to_vector(next_word)
      prompt_vector = vector_add(prompt_vector, new_token_vector)
      encoded_prompt = encode(prompt_vector).to_a.flatten
    end

    completion.join(" ")
  end

  def generate_next_token(latent_vector, method = :deterministic)
    generated_input = decode(latent_vector)
    case method
    when :deterministic then vector_to_single_word(generated_input)
    when :probabilistic then vector_to_single_word_probabilistic(generated_input)
    else raise "Unknown method: #{method}"
    end
  end

  def vector_add(vec1, vec2)
    vec1.zip(vec2).map { |a, b| a + b }
  end

  def save_to_file(filename)
    File.open(filename, "wb") do |file|
      Marshal.dump({
        input_size: @input_size,
        hidden_size: @hidden_size,
        latent_size: @latent_size,
        weights_input_to_hidden: @weights_input_to_hidden,
        weights_hidden_to_latent: @weights_hidden_to_latent,
        weights_latent_to_hidden: @weights_latent_to_hidden,
        weights_hidden_to_output: @weights_hidden_to_output,
        vocabulary: @vocabulary,
        word_to_index: @word_to_index,
      }, file)
    end
    puts "Network saved to #{filename}"
  end

  def self.load_from_file(filename)
    data = File.open(filename, "rb") { |file| Marshal.load(file) }
    nn = new(data[:input_size], data[:hidden_size], data[:latent_size],
             data[:vocabulary], data[:word_to_index])
    %i[weights_input_to_hidden weights_hidden_to_latent weights_latent_to_hidden weights_hidden_to_output].each do |w|
      nn.instance_variable_set("@#{w}", data[w])
    end
    puts "Network loaded from #{filename}"
    nn
  end

  def model_size
    param_count = @weights_input_to_hidden.row_count  * @weights_input_to_hidden.column_count +
                  @weights_hidden_to_latent.row_count * @weights_hidden_to_latent.column_count +
                  @weights_latent_to_hidden.row_count * @weights_latent_to_hidden.column_count +
                  @weights_hidden_to_output.row_count * @weights_hidden_to_output.column_count

    layers = {
      input:   @input_size,
      hidden1: @hidden_size,
      latent:  @latent_size,
      hidden2: @hidden_size,
      output:  @input_size,
    }

    {
      parameters: param_count,
      memory_kb:  (param_count * 4 / 1024.0).round(2),
      layers:     layers,
    }
  end

  private

  # SGD: one example, one update. Accumulator pattern degenerates to
  # "compute and immediately apply".
  def run_sgd(inputs, learning_rate)
    total_loss = 0.0
    inputs.each do |input|
      forward(input)
      grads = backward(input)              # autoencoder: target == input
      total_loss += grads[:loss]
      apply_gradients(grads, learning_rate)
    end
    total_loss
  end

  # Mini-batch / batch: the whole reason this method exists is to show
  # gradient accumulation explicitly.
  #
  #   1. zero an accumulator (one matrix per weight tensor)
  #   2. for each example: forward, backward, ADD its gradient into the accumulator
  #   3. scale by 1/batch_size to get the mean gradient
  #   4. apply once
  #
  # Averaging the *gradients* (not the raw errors) is what makes this
  # mathematically equivalent to taking a step on the mean loss.
  def run_mini_batch(inputs, learning_rate, batch_size)
    total_loss = 0.0
    inputs.each_slice(batch_size) do |batch|
      acc = zero_gradients
      batch.each do |input|
        forward(input)
        grads = backward(input)
        total_loss += grads[:loss]
        accumulate_gradients(acc, grads)
      end
      scale_gradients(acc, 1.0 / batch.size)
      apply_gradients(acc, learning_rate)
    end
    total_loss
  end

  def zero_gradients
    {
      weights_input_to_hidden:  Matrix.zero(@input_size,  @hidden_size),
      weights_hidden_to_latent: Matrix.zero(@hidden_size, @latent_size),
      weights_latent_to_hidden: Matrix.zero(@latent_size, @hidden_size),
      weights_hidden_to_output: Matrix.zero(@hidden_size, @input_size),
    }
  end

  def accumulate_gradients(acc, grads)
    acc[:weights_input_to_hidden]  += grads[:weights_input_to_hidden]
    acc[:weights_hidden_to_latent] += grads[:weights_hidden_to_latent]
    acc[:weights_latent_to_hidden] += grads[:weights_latent_to_hidden]
    acc[:weights_hidden_to_output] += grads[:weights_hidden_to_output]
  end

  def scale_gradients(acc, factor)
    acc[:weights_input_to_hidden]  *= factor
    acc[:weights_hidden_to_latent] *= factor
    acc[:weights_latent_to_hidden] *= factor
    acc[:weights_hidden_to_output] *= factor
  end

  # ReLU: f(x) = max(0, x).  f'(x) = 1 if x > 0 else 0.
  def relu(x)
    x.map { |e| [0, e].max }
  end

  def relu_derivative(e)
    e > 0 ? 1 : 0
  end

  # Sigmoid: f(x) = 1 / (1 + e^-x).  f'(x) = f(x) * (1 - f(x)).
  # When we already have the post-activation `a = sigmoid(x)`, the
  # derivative is just `a * (1 - a)` — which is what backward() uses
  # directly on @output, no extra exp() call needed.
  def sigmoid(x)
    x.map { |e| 1 / (1 + Math.exp(-e)) }
  end
end
