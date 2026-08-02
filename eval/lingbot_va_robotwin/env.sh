#!/usr/bin/env bash
# Single source of truth for the LingBot-VA x RoboTwin 2.0 evaluation pipeline.
# Source this; do not execute it.
#
# Port scheme (deliberately non-overlapping -- see NOTE below):
#   websocket serving port : IWM_WS_PORT_BASE  + gpu_index   -> 29056..29063
#   torch.distributed rdzv : IWM_RDZV_PORT_BASE + gpu_index  -> 29800..29807
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
# once (/home/ubuntu/InstinctWM -> /home/ubuntu/Code/InstinctWM) and a stale IWM_ROOT breaks
# only the arms that import instinctwm -- a broken A/B rather than a broken run.
# BASH_SOURCE is the right variable here: this file is sourced, never executed.
export IWM_ROOT=${IWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/home/ubuntu/RoboTwin}
export LINGBOT_ROOT=${LINGBOT_ROOT:-/home/ubuntu/lingbot-va}

# ---- model ------------------------------------------------------------------
export LINGBOT_CKPT=${LINGBOT_CKPT:-/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin}

# ---- interpreters (two envs, on purpose) ------------------------------------
# The server needs torch 2.9 / diffusers 0.36; the client needs sapien 3.0.0b1 on
# torch 2.4. They are dependency-incompatible, which is exactly why upstream put a
# websocket between them. Never try to merge these.
#
# The server env is now pixi-managed: `[tool.pixi.feature.server]` in pyproject.toml
# pins lingbot-va/requirements.txt and `pixi install -e server` materialises it here.
# Build it with:  pixi install -e server
# The pin lives in pyproject.toml, not in this file, so the lockfile is what makes a
# rerun reproducible -- a hand-rolled venv could not be re-created from the repo.
export IWM_SERVER_PY=${IWM_SERVER_PY:-${IWM_ROOT}/.pixi/envs/server/bin/python}
export IWM_CLIENT_PY=${IWM_CLIENT_PY:-${ROBOTWIN_ROOT}/.venv/bin/python}

if [ ! -x "$IWM_SERVER_PY" ]; then
  echo "WARNING: IWM_SERVER_PY does not exist: $IWM_SERVER_PY" >&2
  echo "         run 'pixi install -e server' from $IWM_ROOT" >&2
fi

# ---- flash-attn import shim -------------------------------------------------
# wan_va/modules/model.py imports flash_attn unconditionally at module scope even
# though the RoboTwin path runs attn_mode='torch'. Set IWM_FA_SHIM=1 to make the
# import-only shim visible. It raises if ever CALLED, so it cannot change numerics.
# Leave unset once a real flash-attn wheel is installed -- PYTHONPATH precedes
# site-packages and would otherwise shadow the real package.
export IWM_FA_SHIM_DIR=${IWM_FA_SHIM_DIR:-/home/ubuntu/iwm_shims}

# ---- ports ------------------------------------------------------------------
export IWM_WS_PORT_BASE=${IWM_WS_PORT_BASE:-29056}
export IWM_RDZV_PORT_BASE=${IWM_RDZV_PORT_BASE:-29800}
export IWM_NUM_GPUS=${IWM_NUM_GPUS:-8}

# ---- run artifacts ----------------------------------------------------------
export IWM_LOG_DIR=${IWM_LOG_DIR:-/home/ubuntu/iwm_logs}
export IWM_RESULT_DIR=${IWM_RESULT_DIR:-/home/ubuntu/iwm_results}
# The server dumps latents/actions/obs tensors here on EVERY chunk via save_async.
# On a full 50-task run this is the largest artifact by far -- keep it on the big disk.
export IWM_VIS_DIR=${IWM_VIS_DIR:-/home/ubuntu/iwm_vis}

mkdir -p "$IWM_LOG_DIR" "$IWM_RESULT_DIR" "$IWM_VIS_DIR"

iwm_ws_port()   { echo $(( IWM_WS_PORT_BASE   + $1 )); }
iwm_rdzv_port() { echo $(( IWM_RDZV_PORT_BASE + $1 )); }

# True if something is already listening on $1.
iwm_port_busy() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1
  fi
}
