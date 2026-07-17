#!/usr/bin/env bash
# Source this to set the paths the build + pipeline use.
#   source scripts/alive2/env.sh
# Override BUILD_ROOT before sourcing to put the ~30GB build elsewhere.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export BUILD_ROOT="${BUILD_ROOT:-$HOME/.cache/t3rl-alive2}"
export LLVM_SRC="$BUILD_ROOT/llvm-project"
export LLVM_BUILD="$LLVM_SRC/build"
export ALIVE2_SRC="$REPO_ROOT/third_party/alive2"
export ALIVE2_BUILD="$ALIVE2_SRC/build"
# The binary our pipeline's alive-harness shells out to:
export ALIVE_TV="$ALIVE2_BUILD/alive-tv"
