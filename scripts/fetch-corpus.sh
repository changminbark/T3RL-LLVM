#!/usr/bin/env bash
# Fetch a C source tree to build the real corpus from (llvm-test-suite/SingleSource).
# Shallow clone (~hundreds MB). Prints the path to pass to build_corpus --src.
#
#   ./scripts/fetch-corpus.sh
#   uv run python -m probe.build_corpus --src "$(./scripts/fetch-corpus.sh -q)" \
#       --out data/corpus/corpus.jsonl --with-mca --max-functions 800
set -euo pipefail

QUIET=0
[ "${1:-}" = "-q" ] && QUIET=1

DEST="${CORPUS_SRC_DIR:-$HOME/.cache/t3rl-corpus/llvm-test-suite}"
SINGLE_SRC="$DEST/SingleSource"

log() { [ "$QUIET" = 1 ] || echo "$@" >&2; }

if [ ! -d "$DEST/.git" ]; then
  log "Cloning llvm-test-suite (shallow) into $DEST ..."
  git clone --depth 1 https://github.com/llvm/llvm-test-suite "$DEST"
else
  log "Already present at $DEST"
fi

if [ ! -d "$SINGLE_SRC" ]; then
  echo "Expected $SINGLE_SRC but it is missing." >&2
  exit 1
fi

log "SingleSource tree ready. Pass this to build_corpus --src :"
echo "$SINGLE_SRC"
