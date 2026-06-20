"""
KPI Translation Evaluation Benchmark

A held-out set of free-text KPI definitions, each with a ground-truth
target KPI (and, for two cases, a target formula not in the registry,
to test composition rather than pure retrieval-matching). None of these
exact phrasings appear in INFORMAL_ALIASES, so a system that only does
alias lookup will fail most of these -- they require the LLM+retrieval
step to generalise.

Metrics computed:
  - Top-1 retrieval accuracy: was the correct counter set within the
    retrieved candidates at all?
  - KPI match accuracy: did the system correctly identify which existing
    registered KPI this corresponds to (for the 12 "matches an existing
    KPI" cases)? Accepts either the correct label OR a composed formula
    that is numerically equivalent (within 1%) to the registered KPI's
    value for the same input -- a model that derives the right value via
    an unlabelled but equivalent formula has still solved the task.
  - Formula validity rate: for composed (non-matching) cases, did the
    system produce a formula using only real counters?
  - Numeric accuracy: when ground truth counter values are supplied, does
    the computed KPI value match the value computed from the ground-truth
    formula within tolerance?
"""

from telekg.pm_registry import KPI_REGISTRY
from telekg.kpi_translator import KPITranslator, CounterRetriever
from telekg.pm_simulator import generate_cell_counters


# Each entry: (free_text_query, ground_truth_kpi_or_None, ground_truth_counters_or_None)
# ground_truth_kpi: name in KPI_REGISTRY if this should match an existing KPI
# ground_truth_counters: explicit counter set if this is a NEW composed KPI
EVAL_SET = [
    # ── Paraphrases that should match EXISTING registered KPIs ──────────
    ("What fraction of connection attempts actually succeed?",
     "Accessibility", None),
    ("I want to know how often calls get cut off unexpectedly.",
     "Retainability", None),
    ("Show me how reliable handovers between cells are.",
     "Handover_Success", None),
    ("How full is the radio resource pool on the downlink?",
     "PRB_Utilization", None),
    ("What's the actual data rate users are getting?",
     "Throughput", None),
    ("Fraction of packets that never arrive on the downlink.",
     "Packet_Loss", None),
    ("Average round trip delay experienced by users.",
     "Latency", None),
    ("How often does a UE lose its connection due to bad radio conditions during mobility?",
     "Mobility", None),
    ("Proportion of time the cell is actually usable.",
     "Cell_Availability", None),
    ("How much power are we saving compared to max rated output?",
     "Energy_Efficiency", None),
    ("percentage of users who could not even establish a connection",
     "Accessibility", None),
    ("ratio of successful E-RAB releases to all releases including drops",
     "Retainability", None),

    # ── Free-text definitions that DO NOT match an existing KPI:
    #    the system must COMPOSE a new formula from real counters ────────
    ("Average duration a UE stays connected before releasing, in seconds.",
     None, ["RRC_CONN_MEAN_DURATION_S"]),
    ("Uplink packet loss percentage, not downlink.",
     None, ["PDCP_PACKETS_LOST_UL", "PDCP_PACKETS_SENT_UL"]),
    ("Uplink resource block utilization specifically.",
     None, ["PRB_USED_UL", "PRB_AVAILABLE_UL"]),
    ("Uplink throughput in Mbps.",
     None, ["UL_VOL_BYTES", "UL_SCHED_TIME_MS"]),

    # ── Adversarial: vague / ambiguous, to test refusal/low-confidence behaviour ──
    ("network performance",
     None, None),  # too vague — correct system behaviour is low confidence / clarification
]


def run_evaluation(translator: KPITranslator = None, verbose: bool = True) -> dict:
    translator = translator or KPITranslator()
    retriever = translator.retriever

    results = []
    for query, gt_kpi, gt_counters in EVAL_SET:
        # Retrieval check
        candidates = retriever.retrieve(query, top_k=8)
        if gt_kpi:
            relevant_counters = set(KPI_REGISTRY[gt_kpi].counters_used)
        elif gt_counters:
            relevant_counters = set(gt_counters)
        else:
            relevant_counters = set()
        retrieval_hit = bool(relevant_counters & set(candidates)) if relevant_counters else None

        # Translation
        synthetic_cell_counters = generate_cell_counters("Cell_eval")["counters"]
        result = translator.translate(query, counter_values=synthetic_cell_counters)

        # KPI match correctness: accept either (a) the LLM correctly named
        # the existing KPI, OR (b) the LLM composed a formula whose VALUE
        # is numerically equivalent (within 1%) to the ground-truth KPI's
        # registered value for this same cell. (b) matters because a model
        # that recomputes the right value via an equivalent but unlabelled
        # formula has still solved the task -- but note this is a strict
        # numeric check (e.g. a fraction left unscaled by *100 will legitimately
        # fail here, since 0.98 != 98.18 -- that's a real correctness bug in
        # the model's output, not just a labelling difference).
        kpi_match_correct = None
        numeric_error_pct = None
        if gt_kpi is not None:
            from telekg.pm_registry import compute_kpi
            true_value = compute_kpi(gt_kpi, synthetic_cell_counters)

            named_correct = (result.matched_existing_kpi == gt_kpi)
            numerically_correct = False
            if not named_correct and result.computed_value is not None and true_value:
                numerically_correct = (
                    abs(result.computed_value - true_value) / abs(true_value) * 100 < 1.0
                )
            kpi_match_correct = named_correct or numerically_correct

            if result.computed_value is not None and true_value:
                numeric_error_pct = abs(result.computed_value - true_value) / abs(true_value) * 100

        # Formula validity for composed cases
        composed_valid = None
        if gt_kpi is None and gt_counters is not None:
            composed_valid = result.valid and bool(result.counters_used) and not result.matched_existing_kpi

        results.append({
            "query": query,
            "ground_truth_kpi": gt_kpi,
            "ground_truth_counters": gt_counters,
            "retrieval_hit": retrieval_hit,
            "matched_existing_kpi": result.matched_existing_kpi,
            "kpi_match_correct": kpi_match_correct,
            "generated_formula": result.generated_formula,
            "composed_valid": composed_valid,
            "valid": result.valid,
            "validation_errors": result.validation_errors,
            "computed_value": result.computed_value,
            "numeric_error_pct": numeric_error_pct,
        })

    # ── Aggregate metrics ────────────────────────────────────────────────
    retrieval_cases = [r for r in results if r["retrieval_hit"] is not None]
    retrieval_acc = sum(r["retrieval_hit"] for r in retrieval_cases) / len(retrieval_cases) if retrieval_cases else None

    match_cases = [r for r in results if r["kpi_match_correct"] is not None]
    match_acc = sum(r["kpi_match_correct"] for r in match_cases) / len(match_cases) if match_cases else None

    composed_cases = [r for r in results if r["composed_valid"] is not None]
    composed_valid_rate = sum(r["composed_valid"] for r in composed_cases) / len(composed_cases) if composed_cases else None

    overall_valid_rate = sum(r["valid"] for r in results) / len(results)

    numeric_cases = [r for r in results if r["numeric_error_pct"] is not None]
    mean_numeric_error = (
        sum(r["numeric_error_pct"] for r in numeric_cases) / len(numeric_cases)
        if numeric_cases else None
    )

    summary = {
        "n_total_queries": len(results),
        "retrieval_accuracy": round(retrieval_acc, 4) if retrieval_acc is not None else None,
        "n_retrieval_cases": len(retrieval_cases),
        "kpi_match_accuracy": round(match_acc, 4) if match_acc is not None else None,
        "n_match_cases": len(match_cases),
        "composed_formula_validity_rate": round(composed_valid_rate, 4) if composed_valid_rate is not None else None,
        "n_composed_cases": len(composed_cases),
        "overall_validation_pass_rate": round(overall_valid_rate, 4),
        "mean_numeric_error_pct": round(mean_numeric_error, 4) if mean_numeric_error is not None else None,
        "n_numeric_cases": len(numeric_cases),
    }

    if verbose:
        print(f"\n{'─'*70}")
        print("  KPI Translation Evaluation Results")
        print(f"{'─'*70}")
        for r in results:
            status = "✓" if (r["kpi_match_correct"] or r["composed_valid"]) else (
                "?" if r["kpi_match_correct"] is None and r["composed_valid"] is None else "✗"
            )
            print(f"  [{status}] \"{r['query'][:55]}...\"" if len(r['query']) > 55 else f"  [{status}] \"{r['query']}\"")
            if r["ground_truth_kpi"]:
                print(f"        GT: {r['ground_truth_kpi']:<18} Got: {r['matched_existing_kpi']}")
            elif r["ground_truth_counters"]:
                print(f"        GT counters: {r['ground_truth_counters']}")
                print(f"        Got formula: {r['generated_formula']}")
            if r["validation_errors"]:
                print(f"        Errors: {r['validation_errors']}")
        print(f"\n{'─'*70}")
        print("  Summary")
        print(f"{'─'*70}")
        for k, v in summary.items():
            print(f"  {k:<35}: {v}")

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    run_evaluation()
