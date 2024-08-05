require "matrix"

class NeuralNetwork
  #
  # This is an autoencoder-style neural network with the following layers:
  # 1. Input layer: Represents the input data
  # 2. First hidden layer: Processes the input data
  # 3. Latent layer: Creates a compressed representation of the data
  # 4. Second hidden layer: Processes the latent representation
  # 5. Output layer: Reconstructs the input data
  #
  #  # Our network structure:
  #     Input -> weights_input_to_hidden -> Hidden (size: hidden_size) -> W2 ->
  #     Latent (size: latent_size) -> W3 ->
  #     Hidden (size: hidden_size) -> W4 -> Output
  #
  # The network also includes an attention mechanism to focus on important features.
  # We currently have a single "attention head".

  attr_reader :vocabulary, :word_to_index, :hidden_layer_1, :hidden_layer_2, :latent_representation

  def initialize(input_size, hidden_size, latent_size, vocabulary = nil, word_to_index = nil) # input_size: The size of the input layer, typically the vocabulary size
    # hidden_size: The size of the hidden layers (both first and second hidden layers)
    # latent_size: The size of the latent layer (compressed representation)
    # vocab: An optional array of words in the vocabulary
    # word_to_index: An optional hash mapping words to their indices in the vocabulary
    @input_size = input_size
    @hidden_size = hidden_size
    @latent_size = latent_size
    @vocabulary = vocabulary
    @word_to_index = word_to_index

    # Initialize weight matrices
    # Each weight matrix connects two layers in the network
    # We use random initialization in the range [-1, 1] for several reasons:
    #
    # 1. Symmetry breaking: Random initialization ensures that neurons in the same layer
    #    start with different weights, allowing them to learn different features.
    #
    # 2. Avoiding saturation: Values between -1 and 1 help prevent neurons from starting
    #    in a saturated state (where the activation function's output is close to its extremes).
    #    This is particularly important for activation functions like sigmoid or tanh.
    #
    # 3. Scale considerations: Starting with relatively small weights helps control the
    #    scale of activations as they propagate through the network, which can help with
    #    the vanishing/exploding gradient problem during training.
    #

    @weights_input_to_hidden = Matrix.build(input_size, hidden_size) { rand(-1.0..1.0) }
    @weights_hidden_to_latent = Matrix.build(hidden_size, latent_size) { rand(-1.0..1.0) }
    @weights_latent_to_hidden = Matrix.build(latent_size, hidden_size) { rand(-1.0..1.0) }
    @weights_hidden_to_output = Matrix.build(hidden_size, input_size) { rand(-1.0..1.0) }
    @weights_attention = Matrix.build(hidden_size, 1) { rand(-1.0..1.0) }
    @hidden_layer_1 = nil
    @hidden_layer_2 = nil
    @latent_representation = nil
  end

  def self.create_from_data(data, hidden_size, latent_size)
    # Creates a new NeuralNetwork instance from the given data
    #
    # Parameters:
    # - data: An array of text samples to train on
    # - hidden_size: The size of the hidden layers
    # - latent_size: The size of the latent layer
    #
    # Returns:
    # - A tuple containing the new NeuralNetwork instance and the input vectors

    # Create a temporary instance to use tokenize_french
    temp_nn = new(1, 1, 1)

    # Extract unique words from the data to create the vocabulary
    words = data.flat_map { |line| temp_nn.tokenize_french(line) }.uniq
    word_to_index = words.each_with_index.to_h

    # Create a new NeuralNetwork instance
    nn = new(words.size, hidden_size, latent_size, words, word_to_index)

    # Convert the text data to input vectors
    vectors = nn.text_to_vectors(data)
    [nn, vectors]
  end

  def text_to_vectors(data)
    # Converts an array of text samples into an array of input vectors
    #
    # Parameters:
    # - data: An array of text samples
    #
    # Returns:
    # - An array of input vectors, where each vector is a binary representation of the text

    data.map { |text| text_to_vector(text) }
  end

  def tokenize_french(text)
    # Tokenizes French text, handling contractions and hyphenated words
    #
    # Parameters:
    # - text: A string of French text to tokenize
    #
    # Returns:
    # - An array of tokens (words)

    # Normalize and downcase the text
    text = text.unicode_normalize(:nfkc).downcase

    # Split on word boundaries, keeping apostrophes within words
    tokens = text.split(/\b/)

    # Process tokens
    processed_tokens = []
    tokens.each_with_index do |token, i|
      next if token.nil? || token.strip.empty?  # Skip nil tokens, empty tokens and whitespace

      if token == "'" && i > 0 && i < tokens.length - 1
        # If it's an apostrophe between two tokens, join them
        processed_tokens[-1] += token + tokens[i + 1].to_s
        tokens[i + 1] = nil  # Mark the next token to be skipped
      elsif token == "-" && i > 0 && i < tokens.length - 1
        # If it's a hyphen between two tokens, join them
        processed_tokens[-1] += token + tokens[i + 1].to_s
        tokens[i + 1] = nil  # Mark the next token to be skipped
      elsif !token.match?(/\A[[:punct:]]+\z/) # Keep tokens that aren't just punctuation
        processed_tokens << token
      end
    end

    # Remove any remaining punctuation at the start or end of tokens
    processed_tokens.map! { |token| token.gsub(/\A[[:punct:]]+|[[:punct:]]+\z/, "") }

    # Remove any empty tokens or whitespace tokens that might have resulted from the punctuation removal
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

  def forward(input)
    # Performs a forward pass through the neural network
    #
    # Parameters:
    # - input: An array representing the input vector
    #
    # Returns:
    # - An array containing [output, h2, attention_scores]
    #   - output: The final output of the network
    #   - h2: The latent representation
    #   - attention_scores: The attention weights applied to the first hidden layer
    #
    # This method implements the following steps:
    # 1. Convert input to a column vector
    # 2. Compute the first hidden layer (hidden_layer_1) using ReLU activation
    # 3. Apply attention mechanism to hidden_layer_1
    # 4. Compute the latent representation (latent_representation) using ReLU activation
    # 5. Compute the second hidden layer (hidden_layer_2) using ReLU activation
    # 6. Compute the final output using sigmoid activation
    #
    # The ReLU derivative is used in the hidden layers, promoting sparse activations and helping with gradient flow in deep networks.
    # The Sigmoid derivative is used in the output layer, providing smooth, bounded outputs suitable for reconstruction tasks or binary classification.

    # Convert input to a column vector
    input_vector = Matrix.column_vector(input)

    # Compute the activations of the first hidden layer
    #
    # 1. Matrix multiplication: @weights_input_to_hidden.transpose * input
    #    - This computes the weighted sum for each neuron in the hidden layer.
    #    - If input has shape (input_size, 1) and @weights_input_to_hidden.transpose has shape (hidden_size, input_size),
    #      the result will have shape (hidden_size, 1).
    # 2. ReLU activation: relu(...)
    # 3. Assignment: @hidden_layer_1 = ...
    #    - Stores the result in @hidden_layer_1, which now contains the activations of the first hidden layer.
    #    - Each element in @hidden_layer_1 represents the output of one neuron in this layer.
    #
    # The resulting @hidden_layer_1 has shape (hidden_size, 1), where each element is the activation
    # of a neuron in the first hidden layer.
    @hidden_layer_1 = relu(@weights_input_to_hidden.transpose * input_vector)

    # Compute attention scores and apply them to create a context vector
    #
    # 1. Compute raw attention scores: @weights_attention.transpose * @hidden_layer_1
    #    - @weights_attention is the weight matrix for the attention mechanism (shape: hidden_size, 1)
    #    - @hidden_layer_1 is the output of the first hidden layer (shape: hidden_size, 1)
    #    - The multiplication results in a scalar value (shape: 1, 1)
    #
    # 2. Apply softmax to get normalized attention scores: softmax(...)
    #    - Softmax converts the raw score into a probability distribution
    #    - In this case, with only one value, it's equivalent to a sigmoid activation
    #    - The result, attention_scores, will be a matrix of shape (1, 1) with a value between 0 and 1
    #
    # Softmax converts a vector of real numbers into a probability distribution.
    # The output values are always in the range (0, 1) and sum up to 1.
    # It emphasizes the largest values and suppresses the smaller ones.

    attention_scores = softmax(@weights_attention.transpose * @hidden_layer_1)

    # 3. Create the context vector: @hidden_layer_1 * attention_scores[0, 0]
    #    - attention_scores[0, 0] extracts the single scalar value from the matrix
    #    - Multiplying @hidden_layer_1 by this scalar effectively scales each element of @hidden_layer_1
    #    - This creates a context vector that emphasizes or de-emphasizes different parts of @hidden_layer_1
    #
    # The attention mechanism allows the network to focus on different parts of the input
    # by learning to assign importance (attention) to different elements of the hidden state.
    # In this simple case with a single attention weight, it learns to either emphasize
    # or de-emphasize the entire hidden state uniformly.
    context_vector = @hidden_layer_1 * attention_scores[0, 0]

    # Compute the latent representation (encoding)
    #
    # 1. Matrix multiplication: @weights_hidden_to_latent.transpose * context
    #    - @weights_hidden_to_latent is the weight matrix connecting the first hidden layer to the latent layer
    #    - @weights_hidden_to_latent.transpose has shape (latent_size, hidden_size)
    #    - context is the output of the attention mechanism, shape (hidden_size, 1)
    #    - The result has shape (latent_size, 1)
    #
    # 2. ReLU activation: relu(...)
    #
    # 3. Assignment: @latent_representation = ...
    #    - Stores the result in @latent_representation, which now contains the latent representation
    #    - Each element in @latent_representation represents one dimension in the latent space
    #
    # The latent representation @latent_representation is a compressed encoding of the input data.
    # It has shape (latent_size, 1), where latent_size is typically smaller than hidden_size.
    # This forces the network to learn a compact, information-dense representation of the input.
    #
    # Key points about the latent representation:
    # - Dimensionality reduction: Compresses the input into a lower-dimensional space
    # - Feature extraction: Each dimension in the latent space may correspond to a learned feature
    # - Information bottleneck: Forces the network to retain only the most important information
    # - Basis for generation: In autoencoders, this representation is used to reconstruct the input
    #
    # The quality and characteristics of this latent representation are crucial for the
    # network's performance in tasks like reconstruction, generation, or classification.

    @latent_representation = relu(@weights_hidden_to_latent.transpose * context_vector)

    # Compute the second hidden layer (start of decoding process)
    #
    # 1. Matrix multiplication: @weights_latent_to_hidden.transpose * @latent_representation
    #    - @weights_latent_to_hidden is the weight matrix connecting the latent layer to the second hidden layer
    #    - @weights_latent_to_hidden.transpose has shape (hidden_size, latent_size)
    #    - @latent_representation is the latent representation, shape (latent_size, 1)
    #    - The result has shape (hidden_size, 1)
    #
    # 2. ReLU activation: relu(...)
    #
    # 3. Assignment: @hidden_layer_2 = ...
    #    - Stores the result in @hidden_layer_2, which contains the activations of the second hidden layer
    #    - Each element in @hidden_layer_2 represents the output of one neuron in this layer
    #
    # The second hidden layer @hidden_layer_2 is the first step in decoding the latent representation.
    # It has shape (hidden_size, 1), typically expanding the data back to a higher dimensionality.
    #
    # Key points about this layer:
    # - Decoding: Begins the process of reconstructing the input from the latent representation
    # - Expansion: Typically increases dimensionality from the latent space
    # - Symmetry: In many autoencoder designs, this layer mirrors the first hidden layer
    # - Feature reconstruction: Each neuron may reconstruct higher-level features of the input
    #
    # The activations in this layer represent an intermediate state between the compact
    # latent representation and the final output. The network learns to progressively
    # reconstruct the input data through this and subsequent layers.

    @hidden_layer_2 = relu(@weights_latent_to_hidden.transpose * @latent_representation)

    # Compute the final output (reconstructed input)
    #
    # 1. Matrix multiplication: @weights_hidden_to_output.transpose * @hidden_layer_2
    #    - @weights_hidden_to_output is the weight matrix connecting the second hidden layer to the output layer
    #    - @weights_hidden_to_output.transpose has shape (input_size, hidden_size)
    #    - @hidden_layer_2 is the second hidden layer activation, shape (hidden_size, 1)
    #    - The result has shape (input_size, 1), matching the original input dimensions
    #
    # 2. Sigmoid activation: sigmoid(...)
    #    - Applies the sigmoid function element-wise: f(x) = 1 / (1 + e^(-x))
    #    - Squashes each output value to the range (0, 1)
    #    - Appropriate for reconstructing normalized input features or probabilities
    #
    # 3. Assignment: @output = ...
    #    - Stores the result in @output, which contains the final reconstructed input
    #    - Each element in @output corresponds to a feature in the original input space
    #
    # The final output @output is the network's attempt to reconstruct the original input.
    # It has shape (input_size, 1), matching the dimensions of the original input.
    #
    # Key points about this output layer:
    # - Reconstruction: In an autoencoder, this represents the reconstructed version of the input
    # - Dimensionality: Matches the original input size, completing the encoding-decoding process
    # - Activation choice: Sigmoid is used here to bound outputs between 0 and 1, which is
    #   suitable if the original inputs were similarly normalized
    # - Loss computation: The difference between this output and the original input
    #   is typically used to compute the reconstruction loss during training
    #
    # The quality of this reconstruction is a key metric for evaluating the performance
    # of the autoencoder. A good reconstruction indicates that the network has successfully
    # learned to compress and decompress the essential features of the input data.

    output = sigmoid(@weights_hidden_to_output.transpose * @hidden_layer_2)

    [output, @latent_representation, attention_scores]
  end

  # The learning rate is a hyperparameter that controls how much the model's weights are adjusted in
  # response to the estimated error each time the model weights are updated.
  # Each epoch is an iteration over the entire dataset. Will increase training time linearly.
  def train(inputs, epochs, learning_rate)
    puts "Training: #{epochs} epochs, learning rate: #{learning_rate}"
    total_start_time = Time.now
    epochs.times do |epoch|
      epoch_start_time = Time.now
      inputs.each do |input|
        output, _, _ = forward(input)
        error = Matrix.column_vector(input) - output
        update_weights(input, error, learning_rate)
      end
      epoch_end_time = Time.now
      epoch_duration = epoch_end_time - epoch_start_time
      puts "Epoch #{epoch + 1}/#{epochs} completed in #{epoch_duration.round(2)} seconds"
    end
    total_end_time = Time.now
    total_duration = total_end_time - total_start_time
    puts "\nTraining completed in #{total_duration.round(2)} seconds"
  end

  def update_weights(input, error, learning_rate)
    # Ensure we have the latest values for hidden layers and latent representation
    forward(input)
    # Updates the weights of the neural network using backpropagation
    #
    # Parameters:
    # - input: The original input vector
    # - error: The difference between the desired output and the actual output
    # - learning_rate: The rate at which the weights should be updated
    #
    # This method implements the following steps:
    # 1. Compute the gradients for each layer
    # 2. Update the weights using the computed gradients

    # Convert input to a column vector
    input_vector = Matrix.column_vector(input)

    # Compute gradients for each layer
    # The error matrix has the same shape as the output of the network, which is
    # (input_size, 1). This means it's a column vector with as many rows as there are
    # elements in the input (or output) of the network.
    # Each element in the error matrix represents the difference between the
    # desired output (which in an autoencoder is the same as the input) and the
    # actual output produced by the network for each input feature.
    # For the output layer, we use the sigmoid derivative
    delta_output = error.map { |e| e * sigmoid_derivative(e) }

    # For the hidden layers, we use the ReLU derivative
    delta_hidden_2 = (@weights_hidden_to_output * delta_output).map.with_index { |e, i| e * relu_derivative(@hidden_layer_2[i, 0]) }
    delta_latent = (@weights_latent_to_hidden * delta_hidden_2).map.with_index { |e, i| e * relu_derivative(@latent_representation[i, 0]) }
    delta_hidden_1 = (@weights_hidden_to_latent * delta_latent).map.with_index { |e, i| e * relu_derivative(@hidden_layer_1[i, 0]) }

    # Update weights
    # The weight update rule is: new_weight = old_weight + learning_rate * (layer_output * delta_of_next_layer)
    @weights_hidden_to_output += learning_rate * (@hidden_layer_2 * delta_output.transpose)
    @weights_latent_to_hidden += learning_rate * (@latent_representation * delta_hidden_2.transpose)
    @weights_hidden_to_latent += learning_rate * (@hidden_layer_1 * delta_latent.transpose)
    @weights_input_to_hidden += learning_rate * (input_vector * delta_hidden_1.transpose)
    @weights_attention += learning_rate * delta_hidden_1
  end

  def encode(input)
    # Encodes the input into a latent representation
    #
    # Parameters:
    # - input: An array representing the input vector
    #
    # Returns:
    # - An array representing the latent encoding of the input
    #
    # This method implements the encoding part of the autoencoder:
    # 1. Convert input to a column vector
    # 2. Compute the first hidden layer (hidden_layer) using ReLU activation
    # 3. Compute the latent representation (weights_hidden_to_latent) using ReLU activation

    # Convert input to a column vector
    input_vector = Matrix.column_vector(input)

    # Compute first hidden layer
    hidden_layer = relu(@weights_input_to_hidden.transpose * input_vector)

    # Compute and return latent representation
    relu(@weights_hidden_to_latent.transpose * hidden_layer).to_a.flatten
  end

  def decode(latent_vector)
    # Decodes a latent representation back into the original input space
    #
    # Parameters:
    # - latent_vector: An array representing the latent encoding
    #
    # Returns:
    # - An array representing the reconstructed input
    #
    # This method implements the decoding part of the autoencoder:
    # 1. Convert latent vector to a column vector
    # 2. Compute the second hidden layer (h3) using ReLU activation
    # 3. Compute the output using sigmoid activation

    # Convert latent vector to a column vector
    latent_matrix = Matrix.column_vector(latent_vector.to_a.flatten)

    # Compute second hidden layer
    hidden_layer = relu(@weights_latent_to_hidden.transpose * latent_matrix)

    # Compute and return output
    output = sigmoid(@weights_hidden_to_output.transpose * hidden_layer)
    output.to_a.flatten
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
    when :deterministic
      vector_to_words(generated_input)
    when :probabilistic
      vector_to_words_probabilistic(generated_input)
    else
      raise "Unknown method: #{method}"
    end
  end

  def generate_completion(prompt, max_tokens = 20, method = :deterministic)
    prompt_vector = text_to_vector(prompt)
    encoded_prompt = encode(prompt_vector).to_a.flatten

    completion = prompt.split

    max_tokens.times do
      # Generate the next token
      adjusted_latent = encoded_prompt.map { |v| v + rand(-0.1..0.1) }
      next_word = generate_next_token(adjusted_latent, method)

      # Break if we've reached an end condition (e.g., end of sentence)
      break if next_word.nil? || next_word.empty?

      # Add the new token to the completion
      completion << next_word

      # Update the prompt vector with the new token
      new_token_vector = text_to_vector(next_word)
      prompt_vector = vector_add(prompt_vector, new_token_vector)
      encoded_prompt = encode(prompt_vector).to_a.flatten
    end

    completion.join(" ")
  end

  def generate_next_token(latent_vector, method = :deterministic)
    generated_input = decode(latent_vector)
    case method
    when :deterministic
      vector_to_single_word(generated_input)
    when :probabilistic
      vector_to_single_word_probabilistic(generated_input)
    else
      raise "Unknown method: #{method}"
    end
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
        weights_attention: @weights_attention,
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
    %i[weights_input_to_hidden weights_hidden_to_latent weights_latent_to_hidden weights_hidden_to_output weights_attention].each do |w|
      nn.instance_variable_set("@#{w}", data[w])
    end
    puts "Network loaded from #{filename}"
    nn
  end

  # for our amusement its interesrting to have some model size
  # indications here.
  def model_size
    # Calculate number of parameters
    param_count = @weights_input_to_hidden.row_count * @weights_input_to_hidden.column_count +
                  @weights_hidden_to_latent.row_count * @weights_hidden_to_latent.column_count +
                  @weights_latent_to_hidden.row_count * @weights_latent_to_hidden.column_count +
                  @weights_hidden_to_output.row_count * @weights_hidden_to_output.column_count +
                  @weights_attention.row_count * @weights_attention.column_count

    # Calculate memory footprint (assuming 4 bytes per float)
    memory_bytes = param_count * 4

    # Layer dimensions
    layers = {
      input: @input_size,
      hidden1: @hidden_size,
      latent: @latent_size,
      hidden2: @hidden_size,
      output: @input_size,
    }

    {
      parameters: param_count,
      memory_kb: (memory_bytes / 1024.0).round(2),
      layers: layers,
    }
  end

  private

  # relu is f(x) = max(0, x)
  # and its derivative is:
  # f'(x) = {
  #   1 if x > 0
  #   0 if x <= 0
  # }
  def relu(x)
    x.map { |e| [0, e].max }
  end

  # The ReLU derivative is crucial in backpropagation:
  #
  # For positive activations, it allows the gradient to pass through unchanged.
  # For negative activations, it completely stops the gradient flow, which can lead to sparse activations and can help with the vanishing gradient problem.

  def relu_derivative(e)
    e > 0 ? 1 : 0
  end

  #  squashes each value to the range (0, 1)
  # f(x) = 1 / (1 + e^(-x))
  # with the derivative: f'(x) = f(x) * (1 - f(x))
  # The derivative of the sigmoid function is expressed in terms of the sigmoid function itself.
  # It reaches its maximum of 0.25 at x = 0 and approaches 0 as x approaches positive or negative infinity.
  def sigmoid(x)
    x.map { |e| 1 / (1 + Math.exp(-e)) }
  end

  #sigmoid_derivative(0)     # => 0.25 (maximum value)
  #sigmoid_derivative(2)     # ≈ 0.105
  #sigmoid_derivative(-2)    # ≈ 0.105
  #sigmoid_derivative(10)    # ≈ very close to 0
  #sigmoid_derivative(-10)   # ≈ very close to 0
  # The Sigmoid derivative is important in backpropagation:

  # It's always positive, which means the gradient direction is solely determined by the error term.
  # It's largest for inputs close to 0, where the sigmoid function is most sensitive to changes.
  # For very large positive or negative inputs, the derivative becomes very small, which can contribute to the vanishing gradient problem in deep networks.
  def sigmoid_derivative(e)
    sigmoid_value = 1 / (1 + Math.exp(-e))
    sigmoid_value * (1 - sigmoid_value)
  end

  # softmax(x_i) = exp(x_i) / sum(exp(x_j)) for j = 1, ..., n
  # It takes a vector of real numbers as input.
  # It applies the exponential function (e^x) to each element.
  # It then normalizes these values by dividing each by the sum of all the exponentials.
  def softmax(x)
    exp_x = x.map { |e| Math.exp(e) }
    sum = exp_x.reduce(:+)
    exp_x.map { |e| e / sum }
  end
end
