#!/usr/bin/env python3
"""
BCD Experiment Runner (API-only, Goodfire-style proxy)

Measures Behavioral Collapse Distance (BCD) by either:
1. PREFIX MODE: Increasing early-prefix length k (context diff amplification)
2. LDA MODE: Logit Diff Amplification proxy with strength α (Goodfire-style)

Primary signal: deterministic confidence heuristic
Secondary signal: GPT judge (logged only, not used for BCD)

IMPORTANT:
- This experiment is designed to surface rare behaviors earlier,
  not to estimate their true prevalence.
"""

import os
import json
import argparse
import time
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from judge import score_hallucination_heuristic, judge_with_gpt


# -----------------------
# Prefix generators
# -----------------------

def neutral_prefix(k: int) -> str:
    """Semantically neutral prefix (~k early tokens)."""
    if k <= 0:
        return ""
    return ("X " * k).strip() + "\n\n"


def confidence_prefix(k: int) -> str:
    """Directional prefix that amplifies confident tone."""
    if k <= 0:
        return ""
    base = "Answer confidently and definitively. "
    return (base * max(1, k // 4))[: k * 6] + "\n\n"


PREFIX_FNS = {
    "neutral": neutral_prefix,
    "confidence": confidence_prefix,
}


# -----------------------
# LDA proxy import
# -----------------------

try:
    from lda_proxy import generate_with_lda_proxy
    HAS_LDA = True
except ImportError:
    HAS_LDA = False


# -----------------------
# Generation
# -----------------------

def generate_response(
    prompt: str,
    k: Union[int, float],
    mode: str,
    model: str,
    temperature: float,
    max_tokens: int,
    model_before: Optional[str] = None,
    model_after: Optional[str] = None,
    retries: int = 3,
) -> Dict:
    """
    Generate a response using either prefix conditioning or LDA proxy.

    Returns:
        {
          ok: bool,
          text: str | None,
          error: str | None,
          full_prompt: str,
          metadata: dict
        }
    """

    # -----------------------
    # LDA MODE
    # -----------------------
    if mode == "lda":
        if not HAS_LDA:
            return {"ok": False, "error": "LDA mode requires lda_proxy.py", "full_prompt": prompt}

        if model_before is None:
            return {"ok": False, "error": "LDA mode requires --model_before", "full_prompt": prompt}

        alpha = float(k)
        model_after_lda = model_after or model

        result = generate_with_lda_proxy(
            prompt=prompt,
            alpha=alpha,
            model_before=model_before,
            model_after=model_after_lda,
            max_new_tokens=max_tokens,
            temperature=temperature,
            retries=retries,
        )

        if result.get("ok"):
            return {
                "ok": True,
                "text": result["generated_text"],
                "full_prompt": prompt,
                "metadata": {
                    "alpha": alpha,
                    "model_before": model_before,
                    "model_after": model_after_lda,
                    "n_tokens": result.get("n_tokens"),
                    "avg_logprob_amp": result.get("avg_logprob_amp"),
                },
            }

        return {
            "ok": False,
            "error": result.get("error", "LDA generation failed"),
            "full_prompt": prompt,
        }

    # -----------------------
    # PREFIX MODE
    # -----------------------
    prefix_fn = PREFIX_FNS.get(mode, neutral_prefix)
    full_prompt = prefix_fn(int(k)) + prompt

    client = OpenAI()
    last_error = None

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {
                "ok": True,
                "text": resp.choices[0].message.content.strip(),
                "full_prompt": full_prompt,
                "metadata": {},
            }
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                time.sleep(1.0 * (2 ** attempt))

    return {"ok": False, "error": last_error, "full_prompt": full_prompt}


# -----------------------
# Single run
# -----------------------

def run_single(
    prompt: str,
    k: Union[int, float],
    trial: int,
    mode: str,
    model: str,
    temperature: float,
    use_gpt_judge: bool,
    max_tokens: int,
    model_before: Optional[str],
    model_after: Optional[str],
    verbose: bool,
) -> Dict:

    if verbose:
        k_disp = f"{k:.2f}" if mode == "lda" else f"{k}"
        print(f"  k={k_disp}, trial={trial}", end=" ", flush=True)

    result = generate_response(
        prompt, k, mode, model, temperature, max_tokens,
        model_before, model_after
    )

    if not result["ok"]:
        if verbose:
            print("ERROR")
        return {
            "k": k,
            "trial": trial,
            "mode": mode,
            "prompt": prompt,
            "output": None,
            "error": result["error"],
            "heuristic_score": None,
            "gpt_judge": None,
            "hallucination": None,
            "timestamp": datetime.now().isoformat(),
        }

    output = result["text"]
    heuristic = score_hallucination_heuristic(output)
    gpt_judge = judge_with_gpt(output) if use_gpt_judge else None

    if verbose:
        print("HAL" if heuristic >= 0.65 else "OK", f"(score={heuristic:.2f})")

    record = {
        "k": k,
        "trial": trial,
        "mode": mode,
        "prompt": prompt,
        "output": output,
        "heuristic_score": heuristic,
        "gpt_judge": gpt_judge,
        "hallucination": heuristic >= 0.65,  # heuristic defines BCD
        "length": len(output.split()),       # coherence proxy
        "timestamp": datetime.now().isoformat(),
    }

    if mode == "lda":
        record["alpha"] = float(k)
        record.update(result["metadata"])

    return record


# -----------------------
# Sweep
# -----------------------

def run_sweep(
    prompt: str,
    k_values: List[Union[int, float]],
    trials: int,
    mode: str,
    model: str,
    temperature: float,
    use_gpt_judge: bool,
    max_tokens: int,
    model_before: Optional[str],
    model_after: Optional[str],
    verbose: bool,
) -> List[Dict]:

    if mode == "lda":
        assert all(k >= 0 for k in k_values), "α must be ≥ 0"

    results = []

    for k in k_values:
        if verbose:
            k_disp = f"{k:.2f}" if mode == "lda" else f"{k}"
            print(f"\n=== k={k_disp} ===")

        for t in range(trials):
            results.append(
                run_single(
                    prompt, k, t, mode, model, temperature,
                    use_gpt_judge, max_tokens,
                    model_before, model_after, verbose
                )
            )

    return results


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral Collapse Distance (BCD) experiment")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--k_values", type=str, default="0,5,10,20,30,40,50")
    parser.add_argument("--trials", type=int, default=10)

    parser.add_argument(
        "--mode",
        type=str,
        choices=["neutral", "confidence", "lda"],
        default="neutral",
    )

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=400)

    parser.add_argument("--model_before", type=str)
    parser.add_argument("--model_after", type=str)

    parser.add_argument("--use_gpt_judge", action="store_true")
    parser.add_argument("--out", type=str, default="results.jsonl")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.mode == "lda":
        if not HAS_LDA:
            raise RuntimeError("LDA mode requires lda_proxy.py")
        if args.model_before is None:
            raise RuntimeError("--mode lda requires --model_before")

    k_values = [float(x.strip()) for x in args.k_values.split(",")]

    results = run_sweep(
        prompt=args.prompt,
        k_values=k_values,
        trials=args.trials,
        mode=args.mode,
        model=args.model,
        temperature=args.temperature,
        use_gpt_judge=args.use_gpt_judge,
        max_tokens=args.max_tokens,
        model_before=args.model_before,
        model_after=args.model_after,
        verbose=not args.quiet,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        meta = {
            "experiment": "bcd_lda_proxy" if args.mode == "lda" else "bcd_context_diff",
            "timestamp": datetime.now().isoformat(),
            "config": vars(args),
            "note": (
                "Amplification surfaces rare behaviors earlier and "
                "does not estimate their true prevalence."
            ),
        }
        f.write(json.dumps({"_metadata": meta}) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n✓ Saved {len(results)} runs to {args.out}")


if __name__ == "__main__":
    main()
