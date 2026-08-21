#!/usr/bin/env bash
# Is this repository releasable right now? Build both artifacts and install each into a clean venv.
#
#   ./scripts/check_release.sh
#
# Release-day surprises are not usually code bugs, they are packaging bugs: a module that only
# resolved because it was in the source tree, a console script that never got an entry point, a data
# file left out of the wheel. None of those fail in development and all of them fail in front of the
# first external user. So the check is deliberately cold -- pip install the built artifact into an
# empty environment and use it from a directory that is not the repository.
#
# Publication itself needs a credential this script does not have and should not want. What it can
# establish is that publication is one command rather than a debugging session.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
echo "=== building sdist + wheel ==="
python3 -m venv "$WORK/build" >/dev/null
"$WORK/build/bin/pip" -q install -U pip build
rm -rf dist
"$WORK/build/bin/python" -m build >/dev/null
ls -1 dist/

echo
echo "=== every module on disk is in the wheel ==="
"$WORK/build/bin/python" - <<'PY'
import os, sys, zipfile, glob
whl = glob.glob("dist/*.whl")[0]
names = set(zipfile.ZipFile(whl).namelist())
disk = [os.path.join(dp, f) for dp, _, fs in os.walk("instinctflash") for f in fs if f.endswith(".py")]
missing = sorted(p for p in disk if p not in names)
print(f"  {len(disk)} modules on disk, {len([n for n in names if n.endswith('.py')])} in the wheel")
if missing:
    print("  MISSING:"); [print("   ", m) for m in missing]
    sys.exit(1)
print("  complete")
PY

for art in dist/*.whl dist/*.tar.gz; do
  echo
  echo "=== cold install: $(basename "$art") ==="
  V="$WORK/$(basename "$art" | tr './' '__')"
  python3 -m venv "$V" >/dev/null
  "$V/bin/pip" -q install -U pip setuptools >/dev/null
  "$V/bin/pip" -q install "$ROOT/$art"
  # run from OUTSIDE the repo, so nothing resolves via the source tree
  ( cd "$WORK" && "$V/bin/python" -c "
import instinctflash, os
assert 'site-packages' in os.path.dirname(instinctflash.__file__), instinctflash.__file__
print('  import OK from', os.path.dirname(instinctflash.__file__).split('site-packages')[-1] or 'site-packages')
print('  public API:', ', '.join(instinctflash.__all__[:5]), '...')
" )
  ( cd "$WORK" && "$V/bin/instinctflash" --help >/dev/null && echo "  console script OK" )
  # informational, and deliberately not allowed to fail the gate: what this line reports is the
  # environment, not the artifact.
  ( cd "$WORK" && "$V/bin/instinctflash" devices 2>&1 | head -1 | sed 's/^/  devices: /' ) || true
done

echo
echo "RELEASABLE: both artifacts build, install cold, and expose a working console script."
echo "Publication needs a credential; nothing else is in the way."
