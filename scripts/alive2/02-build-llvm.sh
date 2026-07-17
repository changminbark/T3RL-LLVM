#!/usr/bin/env bash
# Build LLVM from source with RTTI + exceptions (Alive2 needs this; apt/brew LLVM won't work).
# ~20-30 GB, ~1-2 hr. Idempotent: re-running resumes the ninja build.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

mkdir -p "$BUILD_ROOT"
if [ ! -d "$LLVM_SRC/.git" ]; then
  # Alive2 tracks LLVM main; shallow clone keeps it ~1.5GB instead of ~3GB.
  git clone --depth 1 https://github.com/llvm/llvm-project "$LLVM_SRC"
fi

mkdir -p "$LLVM_BUILD"
cmake -GNinja -S "$LLVM_SRC/llvm" -B "$LLVM_BUILD" \
  -DLLVM_ENABLE_RTTI=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_PROJECTS="llvm;clang" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

ninja -C "$LLVM_BUILD"
echo "LLVM built at $LLVM_BUILD"
