#!/usr/bin/env bash
# Source this to set the paths the build + pipeline use.
#   source scripts/alive2/env.sh
#
# We pin to LLVM 21 because the alive2 submodule is at its v21.0 release tag. This is a
# reproducible released combo (alive2 v21.0 + llvm 21) — do NOT bump one without the other.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"

# Locate LLVM 21: Homebrew keg (macOS) or apt install path (Linux). Override with LLVM21_PREFIX.
if [ -z "${LLVM21_PREFIX:-}" ]; then
  if command -v brew >/dev/null 2>&1 && brew --prefix llvm@21 >/dev/null 2>&1; then
    LLVM21_PREFIX="$(brew --prefix llvm@21)"
  elif [ -d /usr/lib/llvm-21 ]; then
    LLVM21_PREFIX="/usr/lib/llvm-21"
  fi
fi
export LLVM21_PREFIX
export LLVM_BIN="${LLVM21_PREFIX:+$LLVM21_PREFIX/bin}"   # used by probe/tools.py for clang/llc/llvm-mca

export ALIVE2_SRC="$REPO_ROOT/third_party/alive2"
export ALIVE2_BUILD="$ALIVE2_SRC/build"
# The binary our pipeline's alive-harness shells out to:
export ALIVE_TV="$ALIVE2_BUILD/alive-tv"
