"""Report a corpus's mca coverage + bucket balance.

Two different things produce a null `mca_cycles_o3`, and they need different fixes:
  - no `o3_baseline_ir`  -> llvm-extract failed for that function; nothing to score. Expected.
  - has O3 IR, null mca  -> llc/llvm-mca genuinely failed. If this is ~100%, the
    PROBE_TARGET/PROBE_CPU pair is mismatched and the corpus must be rebuilt.

    uv run python scripts/corpus_report.py data/corpus/testsuite-full.jsonl
"""

import json
import sys
from collections import Counter

from probe.schema import CorpusRecord

rows = [CorpusRecord(**json.loads(line)) for line in open(sys.argv[1]) if line.strip()]
n = len(rows)
if not n:
    raise SystemExit("empty corpus")

no_o3 = sum(1 for r in rows if not r.o3_baseline_ir)
null_mca = sum(1 for r in rows if r.mca_cycles_o3 is None)
mca_failed = sum(1 for r in rows if r.o3_baseline_ir and r.mca_cycles_o3 is None)

print(f"records             {n}")
print(f"no o3_baseline_ir   {no_o3:5}  ({no_o3 / n:6.1%})  llvm-extract failed; null mca expected")
print(f"null mca_cycles     {null_mca:5}  ({null_mca / n:6.1%})")
print(f"  had O3, mca fail  {mca_failed:5}  ({mca_failed / n:6.1%})  <- llc/mca failure")
print(f"scored              {n - null_mca:5}  ({(n - null_mca) / n:6.1%})")

print("\nbuckets over SCORED records (the ones usable as reward targets):")
c = Counter((r.size_bucket(), r.has_loops) for r in rows if r.mca_cycles_o3 is not None)
for (bucket, loops), k in sorted(c.items()):
    print(f"  {bucket:>7}  loops={str(loops):5}  {k}")
