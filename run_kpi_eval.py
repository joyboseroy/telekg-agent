"""
run_kpi_eval.py — Run this to get real KPI translation accuracy numbers
using a real LLM (Groq) instead of the offline stub.

Usage:
    # 1. Get a free API key: https://console.groq.com
    export GROQ_API_KEY=gsk_...

    # 2. (Optional) install sentence-transformers for semantic retrieval
    #    — falls back to keyword retrieval if not installed:
    pip install sentence-transformers

    # 3. Run:
    python run_kpi_eval.py
    python run_kpi_eval.py --model llama-3.3-70b-versatile
    python run_kpi_eval.py --no-embeddings   # use keyword retrieval only
    python run_kpi_eval.py --json results.json   # also dump raw results

This prints the same table as `python -m telekg.kpi_eval` but wired to a
REAL Groq-hosted LLM rather than the offline stub, and writes a clean
summary block at the end.
"""

import argparse
import json
import sys

from agents.pipeline import LLMWrapper, DEFAULT_GROQ_MODEL
from telekg.kpi_translator import KPITranslator, CounterRetriever
from telekg.kpi_eval import run_evaluation, EVAL_SET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL, help="Groq model name")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Use keyword retrieval instead of sentence-transformers")
    parser.add_argument("--json", default=None, help="Optional path to dump raw results as JSON")
    args = parser.parse_args()

    print(f"[INFO] Connecting to Groq model: {args.model}")
    llm = LLMWrapper(model=args.model, offline=False)

    if llm.offline:
        print("[ERROR] Could not initialise Groq client (no GROQ_API_KEY found, or "
              "the groq package is not installed: pip install groq).")
        sys.exit(1)

    probe = llm.generate("Reply with exactly one word: OK")
    if not probe or probe.strip().startswith("[STUB]"):
        print(f"[ERROR] Groq did not respond to a test call. Probe response was: {probe!r}")
        print("[ERROR] Refusing to run the evaluation against a non-functional LLM — "
              "these numbers would silently fall back to the offline stub.")
        sys.exit(1)
    print(f"[INFO] Groq responded OK: {probe.strip()[:40]!r}")

    retriever = CounterRetriever(use_embeddings=not args.no_embeddings)
    if args.no_embeddings:
        print("[INFO] Using keyword-overlap retrieval (no sentence-transformers)")
    elif not retriever.use_embeddings:
        print("[WARN] sentence-transformers not available, fell back to keyword retrieval. "
              "Run: pip install sentence-transformers   for semantic retrieval.")
    else:
        print("[INFO] Using sentence-transformers (all-MiniLM-L6-v2) for semantic retrieval")

    translator = KPITranslator(llm=llm, retriever=retriever)

    print(f"\n[INFO] Running evaluation on {len(EVAL_SET)} test queries...\n")
    output = run_evaluation(translator=translator, verbose=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n[INFO] Raw results written to {args.json}")

    s = output["summary"]
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Model used:                          {args.model}")
    print(f"  Retrieval mode:                       {'semantic (sentence-transformers)' if retriever.use_embeddings else 'keyword-overlap'}")
    print(f"  Total test queries:                   {s['n_total_queries']}")
    print(f"  Retrieval accuracy (counter in top-k): {s['retrieval_accuracy']}")
    print(f"  KPI match accuracy (existing KPIs):    {s['kpi_match_accuracy']}  (n={s['n_match_cases']})")
    print(f"  Composed formula validity rate:        {s['composed_formula_validity_rate']}  (n={s['n_composed_cases']})")
    print(f"  Overall validation pass rate:          {s['overall_validation_pass_rate']}")
    print(f"  Mean numeric error vs ground truth:    {s['mean_numeric_error_pct']}%  (n={s['n_numeric_cases']})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
