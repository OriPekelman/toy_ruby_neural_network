# Toy Neural Network Implementation in Ruby with 0 dependencies

This is  an auto-encoder style NN implementation, in Ruby with 0 depenedecies. This is for my own education only. But learning in public is better.

This project implements a simple neural network for processing and generating Mostly French text. Well, my tokenizer has just a couple of somewhat French specificities. Hardly.

If it has already been trained with a specific text file and hyper parametersit will cache the model (with a *.dat file). 

The text file for training is expected to be simple text with newlines.

This is probably the slowest thing on earth. There was a time, a while ago when there were very robust numerical and matrix operations libs in Ruby. But it seems that time has passed?  I only briefly looked. At any rate 0 dependecies is cool :). Even from StdLib we are hardly using anything .. basically `matrix` (oh welll and `optparse`).

## Features
- Text tokenization
- Word embedding
- Autoencoder architecture
- Text generation
  
## Model Characteristics

- Processes single inputs (sentences/phrases) independently
- Uses a simple attention mechanism to focus on different parts of the input
- Creates a compressed latent representation of each input

Some parts of neural_network.rb are heavily documented  - others will probably be at some poinr.

## Usage
Run `ruby run.rb` with optional parameters:
- `--epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate for training (default: 0.01)
- `--corpus`: Input text file (default: simple_french.txt)
- `--prompt`: Starting prompt for text generation (default: "Je")
- `--hidden_size`: Size of hidden layers (default: 4)
- `--latent_size`: Size of latent layer (default: 2)
- `--token_num`: How many tokens to generate (default: 4)

```
 ruby run.rb --epochs 20 --learning_rate 0.005 --hidden_size 8 --latent_size 4 --corpus minimal --prompt="un" --num_tokens 2
```

## Training Methods
The model supports three training methods:
- Stochastic Gradient Descent (SGD): Updates weights after each input (batch_size = 1)
- Mini-batch Gradient Descent: Updates weights after processing a small batch of inputs (1 < batch_size < dataset size)
- Batch Gradient Descent: Updates weights after processing the entire dataset (batch_size = dataset size)

The batch size can be controlled via the --batch_size command-line option.
## Requirements
- Ruby 3.x

This project is for educational purposes only and is not intended for production use.