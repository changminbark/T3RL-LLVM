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

**Notes**
- Corpus IR must come from the **same** LLVM (21). `source scripts/alive2/env.sh` sets `LLVM_BIN`
  so `build_corpus` uses llvm@21's clang. IR from Apple's system clang carries attributes
  (e.g. `frame-pointer=non-leaf-no-reserve`) that alive-tv rejects as "Source file is broken".
- Linux: `01-prereqs.sh` uses `apt` (`clang-21 llvm-21-dev`). Override the LLVM location with
  `LLVM21_PREFIX=/path` before sourcing `env.sh` if it's non-standard.
