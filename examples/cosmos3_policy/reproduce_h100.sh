#!/usr/bin/env bash
# Rerun the README Cosmos3 H100 pairs with the exact protocol.
#
#   Edge (3.86B): 311 -> 186 ms, 1.67x        Nano (15.75B): 482 -> 325 ms, 1.49x
#
#   torch baseline arm : NVIDIA's stock robolab server, verbatim (guardrails no-op'd the way
#                        upstream's own test does), openpi msgpack websocket
#   pipeline arm       : our robotwin policy server, optimization knobs OFF
#   T1 graphs arm      : + --use-cuda-graphs (torch.compile reduce-overhead). This is the
#                        README number on H100 — WITH ITS CAVEAT: single-prompt only, inductor
#                        cudagraph_trees asserts on a prompt change; measured slower on Thor.
#
#   protocol           : canonical policy request — one 540x640 image, [16, 8] action chunk,
#                        4 denoise steps, guidance 1.0; 12 requests x 3 warmup, p50,
#                        byte-identical measure clients per arm.
#
# Needs the PATCHED cosmos-framework checkout's venv (this box: /home/ubuntu/cosmos-framework)
# and the checkpoint in the local HF cache. Arms run strictly one at a time on one idle GPU.
#
#   IFL_COSMOS3_MODEL=edge|nano CUDA_VISIBLE_DEVICES=7 examples/cosmos3_policy/reproduce_h100.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
COSMOS_ROOT="${COSMOS_ROOT:-$HOME/cosmos-framework}"
PY="${IFL_COSMOS3_PY:-$COSMOS_ROOT/.venv/bin/python}"
MODEL="${IFL_COSMOS3_MODEL:-edge}"
PORT="${IFL_COSMOS3_PORT:-29930}"
OUT="${IFL_COSMOS3_OUT:-/tmp/cosmos3_reproduce_h100_$MODEL}"

case "$MODEL" in
  edge) REPO="nvidia/Cosmos3-Edge-Policy-DROID" ;;
  nano) REPO="nvidia/Cosmos3-Nano-Policy-DROID" ;;
  *) echo "IFL_COSMOS3_MODEL must be edge or nano" >&2; exit 2 ;;
esac
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "set CUDA_VISIBLE_DEVICES to one idle GPU (the published pairs are solo-GPU protocols)" >&2
  exit 2
fi
if [ ! -f "$COSMOS_ROOT/cosmos_framework/scripts/action_policy_server_robotwin.py" ]; then
  echo "COSMOS_ROOT ($COSMOS_ROOT) is not the patched cosmos-framework checkout" >&2
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

wait_port() {  # blocks until something listens on $1 or the server died
  while ! "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1', $1))==0 else 1)"; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited before it began serving; see its log" >&2; exit 1
    fi
    sleep 2
  done
}

echo "=== arm 1/3: NVIDIA stock robolab (websocket) ==="
( cd "$COSMOS_ROOT" && PYTHONPATH=. exec "$PY" "$HERE/launch_robolab_stock.py" \
    --checkpoint-path "$CKPT" --action-chunk-size 16 --port "$PORT" ) \
    >"$OUT/stock_server.log" 2>&1 &
SERVER_PID=$!
wait_port "$PORT"
"$PY" "$HERE/measure_openpi_ws.py" --port "$PORT" --label nvidia_stock_pytorch \
    --out "$OUT/stock.json"
stop_server

run_ours() {  # $1 = arm name, $2.. = extra server flags
  local arm="$1"; shift
  echo "=== arm: ours ($arm) ==="
  ( cd "$COSMOS_ROOT" && PYTHONPATH=. exec "$PY" -m cosmos_framework.scripts.action_policy_server_robotwin \
      --checkpoint-path "$CKPT" --port "$PORT" \
      --domain-name droid_lerobot --action-dim 8 --action-chunk-size 16 \
      --expected-image-height 540 --expected-image-width 640 \
      --num-steps 4 --guidance 1.0 "$@" ) >"$OUT/${arm}_server.log" 2>&1 &
  SERVER_PID=$!
  wait_port "$PORT"
  "$PY" "$HERE/measure_predict.py" --port "$PORT" --label "$arm" --out "$OUT/$arm.json"
  stop_server
}

echo "=== arm 2/3: our pipeline (optimization knobs OFF) ==="
run_ours pipeline
echo "=== arm 3/3: our pipeline + CUDA graphs (the README H100 number; single-prompt caveat) ==="
run_ours cudagraphs --use-cuda-graphs

echo
echo "=== $REPO pairs (p50 ms) ==="
"$PY" - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
arms = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
stock = arms["stock"]["wall_ms_p50"]
for name in ("pipeline", "cudagraphs"):
    ours = arms[name]["wall_ms_p50"]
    print(f"stock {stock:.1f} -> {name} {ours:.1f} ms   {stock / ours:.2f}x")
print("README pairs: Edge 310.5 -> 185.8 (1.67x, cudagraphs arm; pipeline 235.7);")
print("              Nano 482.3 -> 324.7 (1.49x, cudagraphs arm; pipeline 327.3)")
print("caveat: the cudagraphs arm is single-prompt only (inductor asserts on prompt change)")
EOF
