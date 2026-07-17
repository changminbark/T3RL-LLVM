#!/usr/bin/env bash
# Build Alive2's alive-tv against the LLVM built by 02-build-llvm.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ ! -f "$LLVM_BUILD/bin/llvm-config" ] && [ ! -d "$LLVM_BUILD/lib/cmake/llvm" ]; then
  echo "LLVM build not found at $LLVM_BUILD. Run 02-build-llvm.sh first." >&2
  exit 1
fi

mkdir -p "$ALIVE2_BUILD"
cmake -GNinja -S "$ALIVE2_SRC" -B "$ALIVE2_BUILD" \
  -DCMAKE_PREFIX_PATH="$LLVM_BUILD" \
  -DBUILD_TV=1 \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "$ALIVE2_BUILD" alive-tv

echo
echo "Done. alive-tv is at: $ALIVE_TV"
echo "Point the pipeline at it:  export ALIVE_TV=$ALIVE_TV"
