# Building Alive2 (`alive-tv`)

The reward oracle. Needs LLVM built from source with RTTI — `apt install llvm` will **not** work.

**Needs ~30 GB free disk and ~1–2 hr.** Linux or macOS.

### Step 0 — get the submodule
```
git submodule update --init --recursive
```

### Step 1 — prerequisites
```
./scripts/alive2/01-prereqs.sh
```

### Step 2 — build LLVM (the slow part)
```
./scripts/alive2/02-build-llvm.sh
```

### Step 3 — build alive-tv
```
./scripts/alive2/03-build-alive2.sh
```

### Step 4 — point the pipeline at it
```
source scripts/alive2/env.sh
"$ALIVE_TV" --version    # sanity check
```
Now `--verifier alive` works: `alive-harness` finds `alive-tv` via `$ALIVE_TV`.

---

**Notes**
- Build goes to `~/.cache/t3rl-alive2` by default. Change it: `export BUILD_ROOT=/path` before Step 2.
- Steps 2 & 3 are idempotent — re-run to resume.
- Alive2 tracks LLVM `main`; if Step 3 fails to compile, the pinned submodule and LLVM `main` drifted — check `third_party/alive2/README.md` for the required LLVM version.
