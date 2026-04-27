# Word-level tokenizer for French text.
# Keeps contractions ("c'est") and hyphenated words ("aujourd'hui") intact.

module Tokenizer
  module_function

  def tokenize_french(text)
    text   = text.unicode_normalize(:nfkc).downcase
    tokens = text.split(/\b/)

    out = []
    tokens.each_with_index do |token, i|
      next if token.nil? || token.strip.empty?

      if (token == "'" || token == "-") && i.between?(1, tokens.length - 2)
        # Glue: previous_token + "'" + next_token  (e.g. "c" + "'" + "est")
        out[-1] += token + tokens[i + 1].to_s
        tokens[i + 1] = nil
      elsif !token.match?(/\A[[:punct:]]+\z/)
        out << token
      end
    end

    out.map { |t| t.gsub(/\A[[:punct:]]+|[[:punct:]]+\z/, "") }
       .reject { |t| t.strip.empty? }
  end
end
