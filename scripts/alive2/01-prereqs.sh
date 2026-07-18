#!/usr/bin/env bash
# Install build prerequisites, including LLVM 21 (matches the alive2 v21.0 submodule pin).
# No LLVM-from-source build needed — the packaged llvm@21 is RTTI-enabled and works.
set -euo pipefail

if command -v brew >/dev/null 2>&1; then
  brew install cmake ninja re2c z3 llvm@21
elif command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y cmake ninja-build re2c z3 libz3-dev clang-21 llvm-21-dev
else
  echo "No brew or apt-get. Install manually: cmake ninja re2c z3 + LLVM 21 (dev headers, RTTI)." >&2
  exit 1
fi
echo "prereqs OK"
