require_relative "transformer"

# Numerical gradient check: verify analytic gradients agree with finite differences.
# For each parameter dimension, set L+(theta) = loss with parameter +eps, L-(theta) = loss with -eps.
# Numerical grad ≈ (L+ - L-) / (2*eps).  Should match the analytic backward output to ~1e-4.

def perturbed_loss(nn, token_ids, parameter_ref, i, j, eps)
  parameter_ref[i][j] += eps
  nn.forward(token_ids)
  loss = nn.backward(token_ids)[:loss]
  parameter_ref[i][j] -= eps
  loss
end

def perturbed_loss_vec(nn, token_ids, parameter_ref, i, eps)
  parameter_ref[i] += eps
  nn.forward(token_ids)
  loss = nn.backward(token_ids)[:loss]
  parameter_ref[i] -= eps
  loss
end

srand(42)

nn = TransformerLM.new(
  vocab_size:     5,
  d_model:        4,
  d_ff:           8,
  n_heads:        2,
  context_length: 4,
  n_layers:       1,
  vocabulary:     %w[a b c d e],
  word_to_index:  { "a" => 0, "b" => 1, "c" => 2, "d" => 3, "e" => 4 },
)

tokens = [0, 1, 2, 3]

nn.forward(tokens)
analytic = nn.backward(tokens)

eps      = 1e-5
mismatches = 0
total      = 0
max_err    = 0.0

# Check a few positions in each parameter type.
checks = [
  ["token_embed",                      nn.instance_variable_get(:@token_embed),       analytic[:token_embed],          :matrix],
  ["pos_embed",                        nn.instance_variable_get(:@pos_embed),         analytic[:pos_embed],            :matrix],
  ["lm_head",                          nn.instance_variable_get(:@lm_head),           analytic[:lm_head],              :matrix],
  ["norm_final_gamma",                 nn.instance_variable_get(:@norm_final_gamma),  analytic[:norm_final_gamma],     :vector],
  ["block_0_w_q[0]",                   nn.instance_variable_get(:@blocks)[0][:w_q][0], analytic[:blocks][0][:w_q][0],  :matrix],
  ["block_0_w_k[1]",                   nn.instance_variable_get(:@blocks)[0][:w_k][1], analytic[:blocks][0][:w_k][1],  :matrix],
  ["block_0_w_v[0]",                   nn.instance_variable_get(:@blocks)[0][:w_v][0], analytic[:blocks][0][:w_v][0],  :matrix],
  ["block_0_w_o",                      nn.instance_variable_get(:@blocks)[0][:w_o],   analytic[:blocks][0][:w_o],      :matrix],
  ["block_0_w_ff1",                    nn.instance_variable_get(:@blocks)[0][:w_ff1], analytic[:blocks][0][:w_ff1],    :matrix],
  ["block_0_w_ff2",                    nn.instance_variable_get(:@blocks)[0][:w_ff2], analytic[:blocks][0][:w_ff2],    :matrix],
  ["block_0_norm1_gamma",              nn.instance_variable_get(:@blocks)[0][:norm1_gamma], analytic[:blocks][0][:norm1_gamma], :vector],
]

checks.each do |name, param, grad, kind|
  case kind
  when :matrix
    rows = [param.length, 2].min
    cols = [param[0].length, 2].min
    rows.times do |i|
      cols.times do |j|
        l_plus  = perturbed_loss(nn, tokens, param, i, j, eps)
        l_minus = perturbed_loss(nn, tokens, param, i, j, -eps)
        nn.forward(tokens)   # restore caches
        nn.backward(tokens)
        numerical = (l_plus - l_minus) / (2 * eps)
        analytic_v = grad[i][j]
        err = (numerical - analytic_v).abs
        max_err = err if err > max_err
        total += 1
        if err > 1e-3
          mismatches += 1
          puts "  MISMATCH #{name}[#{i},#{j}]: numeric=#{numerical.round(6)}  analytic=#{analytic_v.round(6)}  err=#{err.round(6)}"
        end
      end
    end
  when :vector
    [param.length, 4].min.times do |i|
      l_plus  = perturbed_loss_vec(nn, tokens, param, i, eps)
      l_minus = perturbed_loss_vec(nn, tokens, param, i, -eps)
      nn.forward(tokens)
      nn.backward(tokens)
      numerical = (l_plus - l_minus) / (2 * eps)
      analytic_v = grad[i]
      err = (numerical - analytic_v).abs
      max_err = err if err > max_err
      total += 1
      if err > 1e-3
        mismatches += 1
        puts "  MISMATCH #{name}[#{i}]: numeric=#{numerical.round(6)}  analytic=#{analytic_v.round(6)}  err=#{err.round(6)}"
      end
    end
  end
end

puts ""
puts "Checked #{total} parameter positions across #{checks.size} tensors"
puts "Max abs error: #{max_err.round(8)}"
if mismatches.zero?
  puts "✓ Analytic gradients agree with numerical gradients"
else
  puts "✗ #{mismatches} mismatches"
  exit 1
end
