"""Prompt templates for the two generation formats and the cheap ablations.

Ablation knobs (per the plan):
  - include_o3:   also show the -O3 output as a starting point (vs -O0 source only)
  - counterexample: single-turn repair — feed back an Alive2 counterexample for one retry
"""

from __future__ import annotations

from .schema import CorpusRecord, GenFormat

_IR_INSTR = (
    "You are an expert LLVM optimizer. Rewrite the given LLVM IR function so it computes "
    "EXACTLY the same result for every input, but runs faster. Preserve the function signature "
    "and name. Do not change observable behavior. Output ONLY the rewritten LLVM IR in a single "
    "```llvm code block."
)

_C_INSTR = (
    "You are an expert C performance engineer. Rewrite the given C function so it computes "
    "EXACTLY the same result for every input, but runs faster. Preserve the function signature "
    "and name. Do not change observable behavior. Output ONLY the rewritten C in a single "
    "```c code block."
)


def build_prompt(
    record: CorpusRecord,
    fmt: GenFormat,
    *,
    include_o3: bool = False,
    counterexample: str | None = None,
    prior_attempt: str | None = None,
) -> str:
    instr = _IR_INSTR if fmt is GenFormat.ir else _C_INSTR
    lang = "llvm" if fmt is GenFormat.ir else "c"

    parts = [instr, "", "Original function (-O0):", f"```{lang}", record.src_ir.strip(), "```"]

    if include_o3 and record.o3_baseline_ir:
        parts += [
            "",
            "The compiler's -O3 version is below. You must do better than this:",
            f"```{lang}",
            record.o3_baseline_ir.strip(),
            "```",
        ]

    if prior_attempt and counterexample:
        parts += [
            "",
            "Your previous rewrite was WRONG. It differs from the original on this input:",
            "```",
            counterexample.strip(),
            "```",
            "Previous (incorrect) attempt:",
            f"```{lang}",
            prior_attempt.strip(),
            "```",
            "Fix it so it is provably equivalent to the original, and still faster.",
        ]

    return "\n".join(parts)
