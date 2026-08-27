#!/usr/bin/env bash
# Rerun the README DreamZero H100 pair (3227 -> 1843 ms, 1.75x, SCREEN) with the exact protocol.
#
#   torch baseline arm : official serve_dreamzero_wan22.py, shipped configuration
#                        (16 scheduler steps @ CFG 5.0, fixed mask computing 8 DiT forwards)
#   T1 arm             : the SAME server with DYNAMIC_CACHE_SCHEDULE=true — upstream's own
#                        velocity-cosine step skipper. SCREEN TIER: it changes actions by
#                        construction (measured max |dA| 0.288 on identical request streams);
#                        a closed-loop success-rate gate is mandatory before shipping it.
#   protocol           : 12 calls x 3 warmup p50; 3 cameras at 160x320; the first call of the
#                        session warms the causal cache with 1 frame per camera, later calls
#                        send 4. Byte-identical measure client per arm.
#
# Needs the GEAR-Dreams venv (this box: /home/ubuntu/dreamzero-repo/.venv), the checkpoint in
# the local HF cache, and ~75 GB free GPU memory. Arms run strictly one at a time.
#
#   CUDA_VISIBLE_DEVICES=7 examples/dreamzero/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DZ_ROOT="${DREAMZERO_ROOT:-$HOME/dreamzero-repo}"
PY="${IFL_DZ_PY:-$DZ_ROOT/.venv/bin/python}"
PORT="${IFL_DZ_PORT:-29940}"
OUT="${IFL_DZ_OUT:-/tmp/dreamzero_reproduce_h100}"
REPO="GEAR-Dreams/DreamZero-DROID"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pair is a solo-H100 protocol)" >&2
  exit 2
fi
if [ ! -f "$DZ_ROOT/eval_utils/serve_dreamzero_wan22.py" ]; then
  echo "DREAMZERO_ROOT ($DZ_ROOT) is not the GEAR-Dreams checkout" >&2
  exit 2
fi
mkdir -p "$OUT"

CKPT="$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$REPO', local_files_only=True))")"
echo "checkpoint: $CKPT"

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

wait_port() {
  while ! "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', $1))==0 else 1)"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before it began serving; see its log" >&2; exit 1
    fi
    sleep 5
  done
}

run_arm() {  # $1 = arm label, $2 = DYNAMIC_CACHE_SCHEDULE value
  echo "=== arm: $1 (DYNAMIC_CACHE_SCHEDULE=$2) — the ~77 GB load takes minutes ==="
  ( cd "$DZ_ROOT" && DYNAMIC_CACHE_SCHEDULE="$2" exec "$PY" eval_utils/serve_dreamzero_wan22.py \
      --model_path "$CKPT" --port "$PORT" ) >"$OUT/$1_server.log" 2>&1 &
  SERVER_PID=$!
  wait_port "$PORT"
  "$PY" "$HERE/measure_dreamzero.py" --port "$PORT" --label "$1" --out "$OUT/$1.json"
  stop_server
}

run_arm stock false
run_arm stepcache_dynamic true

echo
echo "=== $REPO pair (p50 ms) ==="
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
stock = json.loads((out / "stock.json").read_text())["wall_ms_p50"]
ours = json.loads((out / "stepcache_dynamic.json").read_text())["wall_ms_p50"]
print(f"stock {stock:.1f} -> DYNAMIC_CACHE_SCHEDULE {ours:.1f} ms   {stock / ours:.2f}x")
print("README pair: 3226.7 -> 1843.1 ms (1.75x)")
print("TIER: SCREEN — the skipper changes actions by construction; a closed-loop gate is")
print("mandatory before this arm ships as anyone's default.")
EOF
