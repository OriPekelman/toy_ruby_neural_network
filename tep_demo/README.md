# tep_demo: HTTP API POC via Tep + Spinel

A small HTTP server compiled by Spinel using the
[Tep](https://github.com/OriPekelman/tep) Sinatra-flavoured framework,
demonstrating end-to-end "Ruby source → native binary → fast HTTP" with
our project's FFI inference path slotted in (eventually).

## What works

`hello_api.rb` — minimal "hello world" Tep handler.

```sh
make tep_demo/hello
nohup ./tep_demo/hello > /tmp/h.log 2>&1 &
# server is listening on http://0.0.0.0:4567

# Bench (single Spinel-compiled worker, Connection: close, no
# keep-alive). ~22k req/s peak on this GB10 host:
ruby -rsocket -e '
n = 5000; tn = 4
q = Queue.new; n.times { |i| q << i }
t0 = Time.now
(1..tn).map { Thread.new {
  while !q.empty?
    i = q.pop(true) rescue break
    s = TCPSocket.new("127.0.0.1", 4567)
    s.write("GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    s.read; s.close
  end
}}.each(&:join)
puts "#{n} req @ #{tn} threads: #{(Time.now-t0).round(3)} s (#{(n/(Time.now-t0)).round(0)} req/s)"
'
```

Numbers on gx10 (NVIDIA GB10 / aarch64, 1 Spinel worker):

| Concurrency | Throughput |
|---|---:|
| 1 thread   |   7,745 req/s |
| 4 threads  |  22,054 req/s |
| 16 threads |  19,587 req/s |
| 64 threads |  18,122 req/s |

With keep-alive + tep's prefork (workers > 1) the upstream Tep README
claims ~167k req/s — we use `Connection: close` here for simplicity.

## inference_api.rb — full inference over HTTP

`inference_api.rb` adds three endpoints on top of the hello plumbing:

```
GET /health           ok
GET /model            {"vocab":16,"d_model":32,"d_ff":64,"n_heads":4,"n_layers":2,"context":16}
GET /generate?n=N     {"prompt":[3,7,1],"ids":[3,7,1,1,1,...],"ms":1.46}
```

The handler runs greedy generation through `FullForwardFFICache`
(persistent ggml graph) and reports server-side compute time.

Build + bench:

```sh
make tep_demo/api          # via spinel directly (bypasses tep translator)
nohup ./tep_demo/api > /tmp/api.log 2>&1 &
curl -s "http://localhost:4567/generate?n=5"
```

Numbers on gx10, model = `vocab=16, d_model=32, d_ff=64, n_heads=4,
n_layers=2, T=16`, prompt `[3,7,1]`, 1 worker, Connection: close:

| Request shape       | Threads | Throughput   | Tokens/sec |
|---|---|---:|---:|
| `/generate?n=5`     |   1  |  1,314 req/s |   6,570 |
| `/generate?n=5`     |   4  |  1,758 req/s |   8,790 |
| `/generate?n=5`     |   8  |  1,706 req/s |   8,529 |
| `/generate?n=5`     |  16  |  2,738 req/s |  13,689 |
| `/generate?n=10`    |   4  |    920 req/s |   9,201 |
| `/generate?n=32`*   |   4  |    308 req/s |   9,868 |

\* `n=32` exceeds the realized `T_SEQ=16`, so generation after token 16
gets the wrong context — the bench measures HTTP+forward throughput,
not coherent generation. With KV cache (M2) per-token compute becomes
1 position instead of T_SEQ, projecting an order-of-magnitude
improvement at long contexts.

Built with **Spinel master @ `85a4670`+** (the poly-recv suppression
fix in `4024216` is needed; before that, the build hit a dispatch bug
between `Mat#add` and `Tep::Router#add`). We renamed `Mat#add` →
`Mat#plus` in `lib/transformer.rb` to fully sidestep the arity-mismatch
flavour of that bug — `Mat#add!` (the in-place variant) keeps its
name since method NAMES are what collide.

The `_tep_lib/` directory holds a placeholder-substituted copy of
`~/sites/tep/lib` — needed because our build bypasses tep's translator
(which would normally do the `@TEP_SPHTTP_O@` substitution). Generated
by:

```sh
cp -r ~/sites/tep/lib tep_demo/_tep_lib
sed -i "s|@TEP_SPHTTP_O@|/home/oripekelman/sites/tep/lib/tep/sphttp.o|g" \
    tep_demo/_tep_lib/tep/net.rb
sed -i "s|@TEP_SQLITE_O@|/home/oripekelman/sites/tep/lib/tep/tep_sqlite.o|g" \
    tep_demo/_tep_lib/tep/sqlite.rb
```

## Inference throughput (without the HTTP layer)

`tinynn/full_forward_bench.rb` at vocab=4096, d_model=384, d_ff=1024,
n_heads=6, n_layers=6, T=128:

- Native Ruby:    1179.9 ms / forward
- CPU FFI:           31.0 ms / forward   (38.1× speedup)
- CUDA FFI:          33.7 ms / forward   (33.9× speedup)

So per-token greedy generation without KV cache (M2 work in progress):
roughly 32 tokens/sec on CPU FFI, 30 tokens/sec on CUDA FFI. M2 (KV
cache) would push this to "one forward per new token but only one
position computed" — projected order-of-magnitude faster at long
contexts.
