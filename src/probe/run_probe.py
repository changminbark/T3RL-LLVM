"""Experiment driver: sample K rewrites per function, classify, and report solve@K.

Example (offline, no tools/keys needed):
    uv run python -m probe.run_probe --corpus data/bootstrap --backend mock --k 8 \
        --verifier stub --perf stub

Real run once tools + Person A's harness exist:
    uv run python -m probe.run_probe --corpus data/corpus --backend api --model <id> \
        --k 16 --verifier alive --perf mca
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .backends import make_backend
from .metrics import format_table, summarize
from .outcome import classify
from .prompts import build_prompt
from .schema import CorpusRecord, GenFormat, RewriteResult
from .verifier import make_verifier
from .perf import make_perf


def load_corpus(path: Path) -> dict[str, CorpusRecord]:
    """Load every *.jsonl file under `path` (or the file itself) into id -> CorpusRecord."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records: dict[str, CorpusRecord] = {}
    for f in files:
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = CorpusRecord.model_validate_json(line)
            records[rec.function_id] = rec
    if not records:
        raise SystemExit(f"no corpus records found under {path}")
    return records


def _backend_kwargs(args) -> dict:
    if args.backend == "api":
        return {
            "model": args.model,
            "base_url": args.base_url,
            "api_key_env": args.api_key_env,
            "supports_n": args.supports_n,
        }
    if args.backend == "vllm":
        return {"model": args.model}
    return {}  # mock


def run(args) -> None:
    fmt = GenFormat(args.format)
    records = load_corpus(Path(args.corpus))
    backend = make_backend(args.backend, **_backend_kwargs(args))
    verifier = make_verifier(args.verifier)
    perf = make_perf(args.perf)

    results: list[RewriteResult] = []
    for rec in tqdm(records.values(), desc="functions"):
        prompt = build_prompt(rec, fmt, include_o3=args.include_o3)
        completions = backend.generate(
            prompt, k=args.k, temperature=args.temperature, max_tokens=args.max_tokens
        )
        for i, completion in enumerate(completions):
            results.append(
                classify(rec, completion, i, fmt, verifier, perf, timeout_s=args.timeout)
            )

    summary = summarize(records, results)
    table = format_table(summary)
    print("\n" + table + "\n")
    _write_results(args, records, results, summary, table)


def _write_results(args, records, results, summary, table) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"{args.backend}_{args.verifier}_{args.format}_k{args.k}_{stamp}"

    (out_dir / f"{tag}.rewrites.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in results) + "\n"
    )
    (out_dir / f"{tag}.summary.json").write_text(
        json.dumps({"args": vars(args), "summary": summary}, indent=2)
    )
    (out_dir / f"{tag}.table.txt").write_text(table + "\n")
    print(f"wrote results to {out_dir}/{tag}.*")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 1 Part B — model capability probe")
    p.add_argument("--corpus", required=True, help="JSONL file or dir of *.jsonl")
    p.add_argument("--backend", default="mock", choices=["mock", "api", "vllm"])
    p.add_argument("--model", default="", help="model id (api/vllm)")
    p.add_argument("--base-url", dest="base_url", default="https://api.openai.com/v1")
    p.add_argument("--api-key-env", dest="api_key_env", default="OPENAI_API_KEY")
    # Some OpenAI-compatible servers (e.g. Ollama) ignore `n`; loop K sequential requests instead.
    p.add_argument("--no-supports-n", dest="supports_n", action="store_false",
                   help="provider ignores `n`; sample K completions via K sequential calls")
    p.add_argument("--format", default="ir", choices=["ir", "c"])
    p.add_argument("--verifier", default="stub", choices=["stub", "alive"])
    p.add_argument("--perf", default="stub", choices=["stub", "mca"])
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=2048)
    p.add_argument("--timeout", type=int, default=30, help="verifier timeout (s)")
    p.add_argument("--include-o3", dest="include_o3", action="store_true")
    p.add_argument("--out", default="results")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
