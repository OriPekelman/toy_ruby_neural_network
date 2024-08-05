require_relative "neural_network"
require "test/unit"

puts "Test file loaded successfully"

class TestTokenizeFrench < Test::Unit::TestCase
  def setup
    @nn = NeuralNetwork.new(100, 50, 25)
    puts "NeuralNetwork instance created"
  end

  def test_tokenize_french
    puts "Starting tokenize_french tests"
    test_cases = [
      ["Une über édition. Un prêt", ["une", "über", "édition", "un", "prêt"]],
      ["C'est un test.", ["c'est", "un", "test"]],
      ["L'homme marche dans la rue.", ["l'homme", "marche", "dans", "la", "rue"]],
      ["Qu'est-ce que c'est ?", ["qu'est-ce", "que", "c'est"]],
      ["J'aime les pommes et les poires.", ["j'aime", "les", "pommes", "et", "les", "poires"]],
      ["Il fait beau aujourd'hui !", ["il", "fait", "beau", "aujourd'hui"]],
      ["Parlez-vous français ?", ["parlez-vous", "français"]],
      ["L'oiseau-mouche est petit.", ["l'oiseau-mouche", "est", "petit"]],
      ["Je m'appelle Jean-Pierre.", ["je", "m'appelle", "jean-pierre"]],
      ["Dix-neuf plus vingt-et-un font quarante.", ["dix-neuf", "plus", "vingt-et-un", "font", "quarante"]],
    ]

    test_cases.each_with_index do |(input, expected), index|
      puts "Testing case #{index + 1}: #{input}"
      result = @nn.tokenize_french(input)
      puts "Result: #{result.inspect}"
      assert_equal(expected, result, "Failed for input: #{input}\nExpected: #{expected}\nGot: #{result}")
    end
  end
end

puts "Running tests..."
