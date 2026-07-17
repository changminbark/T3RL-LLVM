#!/usr/bin/env bash
# Install build prerequisites (cmake, ninja, z3, re2c, clang). Linux (apt) or macOS (brew).
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git cmake ninja-build clang re2c z3 libz3-dev python3
elif command -v brew >/dev/null 2>&1; then
  brew install git cmake ninja llvm re2c z3 python3
else
  echo "No apt-get or brew found. Install manually: git cmake ninja clang re2c z3 (+libz3 dev headers)." >&2
  exit 1
fi
echo "prereqs OK"
