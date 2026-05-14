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

## What's blocked: inference_api.rb

`inference_api.rb` — same idea but with `FullForwardFFICache` wired in
behind a `GET /generate?n=N` endpoint. Currently **doesn't build**:
Spinel hits a polymorphic-dispatch issue when transformer.rb's
`Mat#add` and tep's `Tep::Router#add` are both reachable in the same
compilation unit. The generated C dispatch widens
`x_attn.add(ff.out)` to a poly receiver `{Mat, Tep::Router}` and emits
incompatible arms (different arities), producing a `sp_RbVal` that
fails to assign to the typed `LayerCache.x_out` ivar.

Symptom:

```
/tmp/spinel_out.xxx/out.c: In function ‘sp_TransformerLM_transformer_block_into’:
/tmp/spinel_out.xxx/out.c:NNNN: error: incompatible types when assigning to
  type ‘sp_LayerCache *’ {aka ‘struct sp_LayerCache_s *’} from type ‘sp_RbVal’
```

(`iv_x_out` is *also* mis-inferred as `sp_LayerCache *` instead of
`sp_Mat *` — a second pollution downstream of the first.)

Workarounds (pick one):

1. Rename `Mat#add` → `Mat#plus` (or similar) so there's no method-name
   collision with `Tep::Router#add`. ~9 call sites in
   `lib/transformer.rb`; mechanical edit.
2. Bypass Tep's router and roll a tiny HTTP loop directly on top of
   tep's `sphttp.o` parser. Avoids `Tep::Router` entirely.
3. Wait for a Spinel-side narrowing of poly `.method()` receivers
   when call-site types are unambiguous.

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
