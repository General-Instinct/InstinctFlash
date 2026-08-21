#!/usr/bin/env bash
# Task runner. uv has no `pixi run` equivalent, so the task definitions that used to
# live in [tool.pixi.tasks] live here instead -- and the reasons they are pinned the
# way they are travel with them.
#
# THREE ENVIRONMENTS, NOT ONE. Alternating extras inside a single .venv would make
# uv add and remove the whole torch stack on every switch, and `test` would stop
# being a real check the moment torch was left behind in the shared venv. Separate
# directories keep `test` honest and keep the switches instant.
#
#   .venv          no extras -- the one that must stay torch-free
#   .venv-dev      runtime + eval, from uv.lock
#   .venv-server   from eval/lingbot_va_robotwin/server-requirements.txt, a
#                  SEPARATE lock on purpose (see that file's header)
set -euo pipefail

IFL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$IFL_ROOT"

CORE_VENV="$IFL_ROOT/.venv"
DEV_VENV="$IFL_ROOT/.venv-dev"
SERVER_VENV="$IFL_ROOT/.venv-server"
SERVER_IN="$IFL_ROOT/eval/lingbot_va_robotwin/server-requirements.in"
SERVER_TXT="$IFL_ROOT/eval/lingbot_va_robotwin/server-requirements.txt"

usage() {
  cat <<'EOF'
usage: ./scripts/task.sh <task>

  test               core only -- everything needing torch reports SKIP
  test-all           runtime + eval -- torch tests run; diffusers and
                     cosmos_framework ones still SKIP
  gpu-check          report torch / CUDA / visible GPUs

  test-lingbot       the 12-pass run against the real LingBot-VA server
  parity-allocator   the documented 200-cycle allocator sweep
  env-check          print the resolved eval paths

  lock               refresh uv.lock
  lock-server        refresh eval/lingbot_va_robotwin/server-requirements.txt
EOF
}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found on PATH. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 127
fi

# The server env is built from its own compiled lock, not from uv.lock, so that its
# torch pin cannot reach the development environments. `uv pip sync` (not install)
# because the point of the env is to BE the lock -- a stray package left behind from
# an earlier revision is a drifted baseline, which is a mismeasured optimization.
sync_server() {
  [ -f "$SERVER_TXT" ] || {
    echo "missing $SERVER_TXT -- run './scripts/task.sh lock-server' first" >&2
    exit 1
  }
  [ -d "$SERVER_VENV" ] || uv venv --python 3.10 "$SERVER_VENV"
  VIRTUAL_ENV="$SERVER_VENV" uv pip sync "$SERVER_TXT"
  # instinctflash itself is not in that lock (it has no dependencies, and listing it
  # there would put a path dependency in a file that is otherwise pure upstream
  # pins). --no-deps keeps the sync above authoritative.
  VIRTUAL_ENV="$SERVER_VENV" uv pip install --no-deps -e "$IFL_ROOT"
}

case "${1:-}" in
  test)
    UV_PROJECT_ENVIRONMENT="$CORE_VENV" exec uv run python tests/run_tests.py
    ;;

  test-all)
    UV_PROJECT_ENVIRONMENT="$DEV_VENV" exec uv run --extra runtime --extra eval \
      python tests/run_tests.py
    ;;

  gpu-check)
    UV_PROJECT_ENVIRONMENT="$DEV_VENV" exec uv run --extra runtime --extra eval \
      python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpus', torch.cuda.device_count())"
    ;;

  # Server-env tasks source env.sh rather than restating LINGBOT_ROOT / the shim dir.
  # env.sh is the declared single source of truth for eval paths and it already had to
  # survive the tree moving once; a second copy of those paths here is precisely the
  # drift it was written to avoid. Sync BEFORE sourcing, or env.sh warns about the
  # interpreter sync_server is in the middle of creating.
  test-lingbot)
    # The 12-pass run: needs the lingbot-va checkout, so it exercises the real upstream
    # WanAttention allocator rather than a stand-in.
    sync_server
    # shellcheck source=eval/lingbot_va_robotwin/env.sh
    source eval/lingbot_va_robotwin/env.sh
    export PYTHONPATH="$IFL_FA_SHIM_DIR"
    exec "$IFL_SERVER_PY" tests/run_tests.py
    ;;

  parity-allocator)
    # RESULTS.md 7b quotes 200 cycles / 800 checks; the file's own default is 120 / 480.
    # Pin the documented sweep so the published number is the one that gets reproduced.
    sync_server
    # shellcheck source=eval/lingbot_va_robotwin/env.sh
    source eval/lingbot_va_robotwin/env.sh
    export PYTHONPATH="$IFL_FA_SHIM_DIR" CYCLES=200
    exec "$IFL_SERVER_PY" tests/test_ring_allocator.py
    ;;

  env-check)
    # shellcheck source=eval/lingbot_va_robotwin/env.sh
    source eval/lingbot_va_robotwin/env.sh
    echo "IFL_ROOT      $IFL_ROOT"
    echo "LINGBOT_ROOT  $LINGBOT_ROOT"
    echo "ROBOTWIN_ROOT $ROBOTWIN_ROOT"
    echo "LINGBOT_CKPT  $LINGBOT_CKPT"
    echo "IFL_SERVER_PY $IFL_SERVER_PY"
    echo "IFL_CLIENT_PY $IFL_CLIENT_PY"
    ;;

  lock)
    exec uv lock
    ;;

  # --python-version 3.10 pins the solve to the floor in requires-python. Without it
  # the lock would silently describe whatever interpreter happened to run it.
  lock-server)
    exec uv pip compile "$SERVER_IN" \
      --python-version 3.10 \
      --output-file "$SERVER_TXT"
    ;;

  -h | --help | help) usage ;;
  "")
    echo "error: no task given" >&2
    usage >&2
    exit 2
    ;;
  *)
    echo "error: unknown task '$1'" >&2
    usage >&2
    exit 2
    ;;
esac
