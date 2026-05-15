#!/usr/bin/env bash
# Quick throughput bench of tep_demo/openai_api.
#
#   Boot the server, fire N sequential and N concurrent requests with
#   max_tokens=M each, report wall-clock + derived tokens/sec.
#
# Usage:
#   ./tep_demo/bench_api.sh [PORT] [REQUESTS] [TOKENS_PER_REQ] [CONCURRENCY]
# Defaults: PORT=4585, REQUESTS=20, TOKENS=20, CONCURRENCY=4

set -euo pipefail

PORT="${1:-4585}"
REQS="${2:-20}"
TOK="${3:-20}"
CONC="${4:-4}"

if ! pgrep -f "tep_demo/openai_api -p $PORT" >/dev/null 2>&1; then
  echo "starting server on :$PORT (will be killed at end of bench)..."
  ./tep_demo/openai_api -p "$PORT" > /tmp/openai_api_bench.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  # Wait for ready (poll /health)
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 0.2
  done
fi

PAYLOAD=$(jq -nc --arg p "Once upon a time" --argjson n "$TOK" \
  '{model:"gpt2",messages:[{role:"user",content:$p}],max_tokens:$n}')

echo "=== bench: $REQS reqs × $TOK tokens, concurrency=$CONC ==="
echo

echo "--- warm-up (1 request) ---"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
     -H 'Content-Type: application/json' -d "$PAYLOAD" \
     | python3 -c "import json,sys;d=json.load(sys.stdin);print('  →', d['choices'][0]['message']['content'][:60])"
echo

echo "--- sequential ---"
t0=$(date +%s.%N)
for _ in $(seq 1 "$REQS"); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/chat/completions" \
       -H 'Content-Type: application/json' -d "$PAYLOAD"
done
t1=$(date +%s.%N)
elapsed=$(python3 -c "print(f'{${t1}-${t0}:.3f}')")
ms_per_req=$(python3 -c "print(f'{(${t1}-${t0})*1000/${REQS}:.1f}')")
tok_per_sec=$(python3 -c "print(f'{${REQS}*${TOK}/(${t1}-${t0}):.1f}')")
echo "  $REQS reqs in ${elapsed}s   = ${ms_per_req} ms/req   = ${tok_per_sec} tok/s aggregate"
echo

echo "--- concurrent (xargs -P $CONC) ---"
t0=$(date +%s.%N)
seq 1 "$REQS" | xargs -P "$CONC" -I{} curl -s -o /dev/null \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$PAYLOAD"
t1=$(date +%s.%N)
elapsed=$(python3 -c "print(f'{${t1}-${t0}:.3f}')")
ms_per_req=$(python3 -c "print(f'{(${t1}-${t0})*1000/${REQS}:.1f}')")
tok_per_sec=$(python3 -c "print(f'{${REQS}*${TOK}/(${t1}-${t0}):.1f}')")
echo "  $REQS reqs in ${elapsed}s   = ${ms_per_req} ms/req   = ${tok_per_sec} tok/s aggregate"
