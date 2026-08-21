#!/usr/bin/env bash
# Manage the fleet of LingBot-VA policy servers, one per GPU.
#
#   ./servers.sh start [n_gpus]
#   ./servers.sh stop
#   ./servers.sh status
#
# Design rules, each of which exists because of a specific way this has gone wrong:
#
#  1. REFUSE TO LAUNCH ON A BUSY PORT. Silently reusing a port that another process
#     owns means the client talks to the wrong server (or to a server running a
#     different config) and every number after that is garbage.
#  2. START VIA setsid, KILL BY PROCESS GROUP. `torch.distributed.run` forks a child;
#     killing the launcher leaves the real python server holding the port, and the
#     next launch dies on EADDRINUSE while the fleet looks healthy.
#  3. VERIFY EVERY SERVER REACHED "server listening", AND FAIL LOUDLY IF NOT.
#     WebsocketClientPolicy._wait_for_server retries on *any* exception every 5s
#     forever, so a client pointed at a dead port hangs silently rather than erroring.
#     A fleet that is 7/8 up must be treated as down.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

PIDFILE="$IFL_LOG_DIR/server_pgids.txt"

start() {
  local n=${1:-$IFL_NUM_GPUS}

  # -- preflight: every port we intend to use must be free ---------------------
  local busy=0
  for i in $(seq 0 $((n-1))); do
    for p in $(iwm_ws_port "$i") $(iwm_rdzv_port "$i"); do
      if iwm_port_busy "$p"; then
        echo "PREFLIGHT FAIL: port $p (gpu $i) is already in use" >&2
        busy=1
      fi
    done
  done
  if [ "$busy" -ne 0 ]; then
    echo "Refusing to launch. Run './servers.sh stop' first, or pick another port base." >&2
    return 1
  fi

  if [ ! -d "$LINGBOT_CKPT" ]; then
    echo "PREFLIGHT FAIL: LINGBOT_CKPT does not exist: $LINGBOT_CKPT" >&2
    return 1
  fi

  local extra_pypath=""
  if [ "${IFL_FA_SHIM:-0}" = "1" ]; then
    extra_pypath="$IFL_FA_SHIM_DIR"
    echo "NOTE: flash-attn import shim ENABLED ($IFL_FA_SHIM_DIR)."
  fi

  : > "$PIDFILE"
  for i in $(seq 0 $((n-1))); do
    local ws rdzv log
    ws=$(iwm_ws_port "$i"); rdzv=$(iwm_rdzv_port "$i")
    log="$IFL_LOG_DIR/server_gpu$i.log"
    ( cd "$LINGBOT_ROOT" && \
      CUDA_VISIBLE_DEVICES=$i \
      PYTHONPATH="${extra_pypath}" \
      LINGBOT_CKPT="$LINGBOT_CKPT" \
      setsid nohup "$IFL_SERVER_PY" -m torch.distributed.run \
        --nproc_per_node 1 --master_port "$rdzv" \
        wan_va/wan_va_server.py --config-name robotwin \
        --port "$ws" --save_root "$IFL_VIS_DIR" \
        > "$log" 2>&1 & echo $! >> "$PIDFILE" )
    echo "gpu$i -> ws:$ws rdzv:$rdzv log:$log"
  done

  # -- wait for all of them, and insist on ALL ---------------------------------
  echo "waiting for $n servers to listen (timeout 600s)..."
  local deadline=$(( SECONDS + 600 ))
  while [ $SECONDS -lt $deadline ]; do
    local up=0
    for i in $(seq 0 $((n-1))); do
      iwm_port_busy "$(iwm_ws_port "$i")" && up=$((up+1))
    done
    if [ "$up" -eq "$n" ]; then
      echo "all $n servers listening."
      status
      return 0
    fi
    # fail fast if a server process died
    for i in $(seq 0 $((n-1))); do
      if grep -qE "Traceback|ChildFailedError|CUDA out of memory" "$IFL_LOG_DIR/server_gpu$i.log" 2>/dev/null; then
        echo "SERVER gpu$i FAILED -- see $IFL_LOG_DIR/server_gpu$i.log" >&2
        grep -E "Error|error while attempting|out of memory" "$IFL_LOG_DIR/server_gpu$i.log" | head -5 >&2
        return 1
      fi
    done
    sleep 5
  done
  echo "TIMEOUT waiting for servers." >&2
  status
  return 1
}

stop() {
  # kill by process group; see rule 2
  if [ -f "$PIDFILE" ]; then
    while read -r pid; do
      [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null
    done < "$PIDFILE"
  fi
  # Match BOTH the stock launcher and serve_variant.py. Matching only the former left
  # A/B variant servers holding their ports, which the next start correctly refused --
  # a loud failure, but only because the preflight check exists.
  # NEVER kill our own process tree. `pkill -f serve_variant.py` matches ANY command line
  # containing that string -- including the caller's, if the caller also mentions it (e.g. a
  # shell that stops the fleet and relaunches it in one line). That kills the operator's shell
  # mid-script, which has now happened three times. Exclude our own PID, our parent, and our
  # process group, and only ever match python processes.
  local self=$$ parent=$PPID mypg
  mypg=$(ps -o pgid= -p $$ | tr -d ' ')
  _iwm_kill() {  # _iwm_kill <signal> <pattern>
    local sig=$1 pat=$2 pid pg
    for pid in $(pgrep -f "$pat" 2>/dev/null); do
      [ "$pid" = "$self" ] && continue
      [ "$pid" = "$parent" ] && continue
      pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ "$pg" = "$mypg" ] && continue
      case "$(ps -o comm= -p "$pid" 2>/dev/null)" in python*) ;; *) continue ;; esac
      kill "-$sig" "$pid" 2>/dev/null
    done
  }
  for pat in "wan_va/wan_va_server.py" "serve_variant.py"; do _iwm_kill TERM "$pat"; done
  sleep 3
  for pat in "wan_va/wan_va_server.py" "serve_variant.py"; do _iwm_kill 9 "$pat"; done
  # Anything still holding a serving port is killed by port, so `stop` is authoritative.
  for i in $(seq 0 $((IFL_NUM_GPUS-1))); do
    p=$(iwm_ws_port "$i")
    for pid in $(ss -ltnp "sport = :${p}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u); do
      kill -9 "$pid" 2>/dev/null
    done
  done
  sleep 2
  rm -f "$PIDFILE"
  echo "stopped."
}

status() {
  printf "%-6s %-8s %-10s %s\n" GPU PORT LISTENING LOG_TAIL
  for i in $(seq 0 $((IFL_NUM_GPUS-1))); do
    local ws state tail_
    ws=$(iwm_ws_port "$i")
    if iwm_port_busy "$ws"; then state=yes; else state=NO; fi
    tail_=$(grep -h "server listening on" "$IFL_LOG_DIR/server_gpu$i.log" 2>/dev/null | tail -1 | sed 's/.*INFO - //')
    printf "%-6s %-8s %-10s %s\n" "$i" "$ws" "$state" "${tail_:-<none>}"
  done
}

case "${1:-status}" in
  start)  shift; start "$@" ;;
  stop)   stop ;;
  status) status ;;
  *) echo "usage: $0 {start [n_gpus]|stop|status}" >&2; exit 2 ;;
esac
