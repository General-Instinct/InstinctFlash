#!/usr/bin/env bash
# Single source of truth for the LingBot-VA x RoboTwin 2.0 evaluation pipeline.
# Source this; do not execute it.
#
# Port scheme (deliberately non-overlapping -- see NOTE below):
#   websocket serving port : IFL_WS_PORT_BASE  + gpu_index   -> 29056..29063
#   torch.distributed rdzv : IFL_RDZV_PORT_BASE + gpu_index  -> 29800..29807
#
# NOTE: the upstream launch scripts use START_PORT=29056 and MASTER_PORT=29061,
# i.e. the rendezvous port of GPU 0 collides with the websocket port of GPU 5
# once you fan out to 8 GPUs. That collision does not fail loudly at the fleet
# level -- 7 servers come up, one dies, and a client pointed at the dead port
# blocks forever in WebsocketClientPolicy._wait_for_server (it retries on *any*
# exception, every 5s, silently). Keep the two ranges far apart.

set -u

# ---- repos ------------------------------------------------------------------
# Derived from this file's own location rather than written down, because the tree has moved
# once (/home/ubuntu/InstinctFlash -> /home/ubuntu/Code/InstinctFlash) and a stale IFL_ROOT breaks
# only the arms that import instinctflash -- a broken A/B rather than a broken run.
# BASH_SOURCE is the right variable here: this file is sourced, never executed.
export IFL_ROOT=${IFL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/home/ubuntu/RoboTwin}
export LINGBOT_ROOT=${LINGBOT_ROOT:-/home/ubuntu/lingbot-va}

# ---- model ------------------------------------------------------------------
export LINGBOT_CKPT=${LINGBOT_CKPT:-/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin}

# ---- interpreters (two envs, on purpose) ------------------------------------
# The server needs torch 2.9 / diffusers 0.36; the client needs sapien 3.0.0b1 on
# torch 2.4. They are dependency-incompatible, which is exactly why upstream put a
# websocket between them. Never try to merge these.
#
# A lock for the server env now exists -- server-requirements.in next to this file,
# compiled to server-requirements.txt. It is deliberately not part of pyproject.toml:
# uv builds one universal lockfile, and sharing it would pin the development envs to
# the server's torch. That header explains it in full. Build it with:
#                 ./scripts/task.sh test-lingbot   (which syncs it first)
#          or:    uv venv .venv-server --python 3.10 \
#                   && VIRTUAL_ENV=.venv-server uv pip sync \
#                        eval/lingbot_va_robotwin/server-requirements.txt
#
# BUT THE DEFAULT BELOW IS STILL THE HAND-ROLLED VENV, ON PURPOSE.
# /home/ubuntu/.venv-lingbot is the environment every frozen number was measured in --
# P001's 2.11x through the 3.38x episode-mode chain -- including the deliberate removal
# of flash-attn. A venv rebuilt from the lock is equivalent only if the lock reproduces
# it exactly, and that is an empirical claim we have not tested. Flipping the default
# first and checking later would silently re-measure the whole chain against a different
# substrate.
#
# To migrate: build .venv-server, run probe_bitexact (max|delta action| = 0) and one
# probe_episode arm against .venv-lingbot, and only then change this line. The lock is
# the right destination -- a hand-rolled venv cannot be re-created from the repo -- but
# the parity check comes before the switch, not after.
export IFL_SERVER_PY=${IFL_SERVER_PY:-/home/ubuntu/.venv-lingbot/bin/python}
export IFL_CLIENT_PY=${IFL_CLIENT_PY:-${ROBOTWIN_ROOT}/.venv/bin/python}

if [ ! -x "$IFL_SERVER_PY" ]; then
  echo "WARNING: IFL_SERVER_PY does not exist: $IFL_SERVER_PY" >&2
  echo "         run './scripts/task.sh test-lingbot' from $IFL_ROOT, or see the" >&2
  echo "         build command in $(basename "${BASH_SOURCE[0]}")" >&2
fi

# ---- flash-attn import shim -------------------------------------------------
# wan_va/modules/model.py imports flash_attn unconditionally at module scope even
# though the RoboTwin path runs attn_mode='torch'. Set IFL_FA_SHIM=1 to make the
# import-only shim visible. It raises if ever CALLED, so it cannot change numerics.
# Leave unset once a real flash-attn wheel is installed -- PYTHONPATH precedes
# site-packages and would otherwise shadow the real package.
export IFL_FA_SHIM_DIR=${IFL_FA_SHIM_DIR:-/home/ubuntu/iwm_shims}

# ---- ports ------------------------------------------------------------------
export IFL_WS_PORT_BASE=${IFL_WS_PORT_BASE:-29056}
export IFL_RDZV_PORT_BASE=${IFL_RDZV_PORT_BASE:-29800}
export IFL_NUM_GPUS=${IFL_NUM_GPUS:-8}

# ---- run artifacts ----------------------------------------------------------
export IFL_LOG_DIR=${IFL_LOG_DIR:-/home/ubuntu/iwm_logs}
export IFL_RESULT_DIR=${IFL_RESULT_DIR:-/home/ubuntu/iwm_results}
# The server dumps latents/actions/obs tensors here on EVERY chunk via save_async.
# On a full 50-task run this is the largest artifact by far -- keep it on the big disk.
export IFL_VIS_DIR=${IFL_VIS_DIR:-/home/ubuntu/iwm_vis}

mkdir -p "$IFL_LOG_DIR" "$IFL_RESULT_DIR" "$IFL_VIS_DIR"

iwm_ws_port()   { echo $(( IFL_WS_PORT_BASE   + $1 )); }
iwm_rdzv_port() { echo $(( IFL_RDZV_PORT_BASE + $1 )); }

# True if something is already listening on $1.
iwm_port_busy() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1
  fi
}
