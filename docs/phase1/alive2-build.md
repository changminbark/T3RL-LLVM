# Building Alive2 (`alive-tv`)

The reward oracle. We pin the alive2 submodule to its **v21.0** release tag and build against
**LLVM 21**, a reproducible released combo — no LLVM-from-source build needed. ~15 min, a few GB.

> Why pinned: alive2's `main` tracks LLVM *main* and won't compile against any released LLVM
> (we hit both `DenormalFPEnv` missing on LLVM 21 and `Triple`→`StringRef` removal on LLVM 22).
> The v21.0 tag + `llvm@21` is the last known-good released pairing. Bump both together or neither.

### Step 0 — get the submodule
```
git submodule update --init --recursive
```

### Step 1 — prerequisites (installs llvm@21, z3, re2c, cmake, ninja)
```
./scripts/alive2/01-prereqs.sh
```

### Step 2 — build alive-tv
```
./scripts/alive2/02-build-alive2.sh
```

### Step 3 — point the pipeline at it
```
source scripts/alive2/env.sh     # exports ALIVE_TV and LLVM_BIN (-> llvm@21)
"$ALIVE_TV" --version            # sanity check
```
Now `--verifier alive` works: `alive-harness` finds `alive-tv` via `$ALIVE_TV`, and the corpus
builder + perf scorer use the same LLVM 21 via `$LLVM_BIN`.

---

## Uninstalling (reclaiming ~5.4 GB)

Almost all the weight is Homebrew's `llvm@21` (5.0 GB), not alive2 (18 MB). Removing it means
`--verifier alive` stops working and corpora can't be regenerated to match the oracle; existing
`data/corpus/*.jsonl` stay readable. Rebuilding later is Steps 0–3 again.

```bash
# 1. Drop the built binaries + submodule checkout (keeps the .gitmodules pin, so Step 0 restores it)
git submodule deinit -f third_party/alive2
rm -rf .git/modules/third_party/alive2      # deinit leaves this ~7 MB behind

# 2. Uninstall the toolchain this build owns exclusively
brew uninstall llvm@21 re2c                 # 5.0 GB + 14 MB
brew cleanup llvm@21                        # the cached 349 MB bottle
```

**Do not remove** `z3` (Homebrew's `llvm` formula depends on it — check with
`brew uses --installed z3`), nor `cmake` / `ninja` / `zstd`. Verify before uninstalling anything
else: `brew uses --installed <formula>` must print nothing.

`env.sh` exports are per-shell, so nothing persists after you close the terminal — unless you added
`source scripts/alive2/env.sh` to `~/.zshrc`, in which case remove that line too.

Removing the submodule from the *repo* (rather than just this machine) is a separate, shared change
— `git rm third_party/alive2` plus deleting the `.gitmodules` stanza and `scripts/alive2/`. Prefer
the local removal above so the build stays one command away for everyone else.

---

**Notes**
- Corpus IR must come from the **same** LLVM (21). `source scripts/alive2/env.sh` sets `LLVM_BIN`
  so `build_corpus` uses llvm@21's clang. IR from Apple's system clang carries attributes
  (e.g. `frame-pointer=non-leaf-no-reserve`) that alive-tv rejects as "Source file is broken".
- Linux: `01-prereqs.sh` uses `apt` (`clang-21 llvm-21-dev`). Override the LLVM location with
  `LLVM21_PREFIX=/path` before sourcing `env.sh` if it's non-standard.
