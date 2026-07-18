#!/usr/bin/env bash
# Build alive2's alive-tv against the packaged LLVM 21 (~15 min, no from-source LLVM build).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ -z "${LLVM21_PREFIX:-}" ]; then
  echo "LLVM 21 not found. Run ./scripts/alive2/01-prereqs.sh first (installs llvm@21)." >&2
  exit 1
fi
Z3_PREFIX="$( (command -v brew >/dev/null 2>&1 && brew --prefix z3) || echo /usr )"

mkdir -p "$ALIVE2_BUILD"
cmake -GNinja -S "$ALIVE2_SRC" -B "$ALIVE2_BUILD" \
  -DCMAKE_PREFIX_PATH="$LLVM21_PREFIX;$Z3_PREFIX" \
  -DBUILD_TV=1 -DCMAKE_BUILD_TYPE=Release

ninja -C "$ALIVE2_BUILD" alive-tv

echo
echo "Done. alive-tv: $ALIVE_TV"
"$ALIVE_TV" --version
echo "Pipeline env is set by:  source scripts/alive2/env.sh   (exports ALIVE_TV + LLVM_BIN)"
