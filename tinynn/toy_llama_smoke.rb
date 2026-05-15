# tinynn/toy_llama_smoke.rb — smoke for Toy::RMSNorm + Toy::SwiGLU.
#
# Verifies the new llama-family building blocks compile under Spinel
# and produce correct math on tiny inputs. No GGUF, no real weights.
#
# Spinel gotcha: don't name a local `rms` — TransformerLM#rms_norm_backward
# has a `rms` Array<Float> parameter. Two types under the same local
# name unify to sp_RbVal and break the existing backward code.

require_relative "../lib/toy"

# ---- Toy::RMSNorm ----
puts "=== Toy::RMSNorm ==="
rnorm = Toy::RMSNorm.new(4)
xn    = Mat.new(2, 4)
# Row 0: [1, 1, 1, 1]   → mean(x^2)=1 → y = [1, 1, 1, 1] (gamma=1)
# Row 1: [1, 2, 3, 4]   → mean(x^2)=7.5 → y = x / sqrt(7.5)
xn.flat[0]=1.0; xn.flat[1]=1.0; xn.flat[2]=1.0; xn.flat[3]=1.0
xn.flat[4]=1.0; xn.flat[5]=2.0; xn.flat[6]=3.0; xn.flat[7]=4.0
yn = rnorm.call(xn)
puts "  y[0] ≈ [1, 1, 1, 1]: [" + yn.flat[0].to_s + ", " +
     yn.flat[1].to_s + "]"
puts "  y[1,0] = " + yn.flat[4].to_s
puts "  expected y[1,0] = " + (1.0 / Math.sqrt(7.5)).to_s

# ---- Toy.silu! ----
puts "=== Toy.silu! ==="
ms = Mat.new(1, 3)
ms.flat[0] = 0.0
ms.flat[1] = 1.0
ms.flat[2] = -1.0
Toy.silu!(ms)
puts "  silu([0, 1, -1]) = [" + ms.flat[0].to_s + ", " +
     ms.flat[1].to_s + ", " + ms.flat[2].to_s + "]"
puts "  expected:          [0, 0.7311, -0.2689]"

# ---- Toy.hadamard! ----
puts "=== Toy.hadamard! ==="
mh_dst = Mat.new(1, 3); mh_dst.flat[0]=2.0; mh_dst.flat[1]=3.0; mh_dst.flat[2]=4.0
mh_src = Mat.new(1, 3); mh_src.flat[0]=0.5; mh_src.flat[1]=2.0; mh_src.flat[2]=0.0
Toy.hadamard!(mh_dst, mh_src)
puts "  [2,3,4] * [0.5,2,0] = [" + mh_dst.flat[0].to_s + ", " +
     mh_dst.flat[1].to_s + ", " + mh_dst.flat[2].to_s + "]"
puts "  expected:             [1, 6, 0]"

# ---- Toy::SwiGLU end-to-end ----
puts "=== Toy::SwiGLU ==="
swi = Toy::SwiGLU.new(8, 16)   # d_model=8, d_ff=16
swi.w_gate.fill_random(0.1)
swi.w_up.fill_random(0.1)
swi.w_down.fill_random(0.1)
swi_in = Mat.new(3, 8)
swi_in.fill_random(1.0)
swi_out = swi.call(swi_in)
puts "  out shape: [" + swi_out.nrows.to_s + ", " + swi_out.ncols.to_s + "]"
puts "  (expected [3, 8])"
puts "  out[0,0] = " + swi_out.flat[0].to_s

puts "ok"
