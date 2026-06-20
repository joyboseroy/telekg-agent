"""
TeleKG-Agent Demo Runner

Demonstrates the full Knowledge Plane pipeline:

  1. Generates synthetic network (100 cells, 10 KPIs, 8 features, 20 events)
  2. Builds FalkorDB Knowledge Graph (or dry-runs if FalkorDB unavailable)
  3. Selects a SW_UPGRADE change event
  4. Runs the 5-agent pipeline
  5. Prints a structured report showing:
     - Impact analysis (features/KPIs/cells at risk)
     - Root cause narrative
     - Generated test cases
  6. Runs baseline comparison: static test selection vs. KG-guided dynamic selection

Usage:
    python main.py                  # Full run (needs FalkorDB + GROQ_API_KEY)
    python main.py --dry-run        # Graph and LLM stubs (no deps needed)
    python main.py --dry-run --eval # Run evaluation comparison
"""

import json
import argparse
import pathlib
import sys
import os

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from telekg.simulator import build_full_dataset
from telekg.graph_builder import build_graph
from telekg.reasoner import TeleKGReasoner
from agents.pipeline import LLMWrapper, run_pipeline_sequential, build_pipeline

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         TeleKG-Agent: Telecom Knowledge Plane Demo              ║
║   Knowledge Graph + Multi-Agent Dynamic Test Generation          ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_section(title: str):
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")


def run_demo(dry_run=False, run_eval=False):
    print(BANNER)

    # ── Step 1: Generate synthetic network data ────────────────────────
    print_section("Step 1: Generating Synthetic Telecom Network")
    data_file = pathlib.Path("data/synthetic_network.json")
    if not data_file.exists():
        data = build_full_dataset()
        data_file.parent.mkdir(exist_ok=True)
        data_file.write_text(json.dumps(data, indent=2))
    else:
        data = json.loads(data_file.read_text())

    print(f"  ✓ Cells:           {len(data['cells'])}")
    print(f"  ✓ KPI snapshots:   {len(data['kpis'])}")
    print(f"  ✓ Change events:   {len(data['change_events'])}")
    print(f"  ✓ Test cases:      {len(data['test_cases'])}")
    print(f"  ✓ KPI causal edges:{len(data['kpi_causal_graph'])}")

    # ── Step 2: Build Knowledge Graph ─────────────────────────────────
    print_section("Step 2: Building FalkorDB Knowledge Graph")
    if dry_run:
        print("  [DRY-RUN] Skipping FalkorDB — using mock reasoner")
        reasoner = TeleKGReasoner(dry_run=True)
    else:
        builder = build_graph(dry_run=False)
        reasoner = TeleKGReasoner(graph=builder.graph)

    # ── Step 3: Select a change event ─────────────────────────────────
    print_section("Step 3: Selecting Change Event to Process")
    sw_events = [e for e in data["change_events"] if e["event_type"] == "SW_UPGRADE"]
    event = sw_events[0]
    print(f"  Event ID:    {event['event_id']}")
    print(f"  Type:        {event['event_type']}")
    print(f"  Trigger:     {event['trigger']}")
    print(f"  Cells:       {len(event['affected_cells'])} affected")
    print(f"  Description: {event['description']}")

    # ── Step 4: Run multi-agent pipeline ──────────────────────────────
    print_section("Step 4: Running Knowledge Plane Agent Pipeline")
    llm = LLMWrapper(offline=dry_run)

    try:
        pipeline = build_pipeline(reasoner, llm)
        if pipeline:
            final_state = pipeline.invoke({
                "event": event,
                "impact": {},
                "root_cause_summary": "",
                "generated_tests": [],
                "test_results": [],
                "agent_log": [],
            })
        else:
            raise ImportError("LangGraph not available")
    except Exception as e:
        print(f"  [INFO] LangGraph not available ({e}), using sequential runner")
        final_state = run_pipeline_sequential(event, reasoner, llm)

    # ── Step 5: Print agent log ────────────────────────────────────────
    print_section("Agent Pipeline Execution Log")
    for entry in final_state.get("agent_log", []):
        print(f"  {entry}")

    # ── Step 6: Print impact analysis ─────────────────────────────────
    impact = final_state.get("impact", {})
    print_section("Impact Analysis Results (from Knowledge Graph Traversal)")
    print(f"  Release:          {impact.get('release', 'N/A')}")
    print(f"  Features touched: {', '.join(impact.get('features', []))}")
    print(f"  Direct KPIs:      {', '.join(impact.get('direct_kpis', []))}")
    print(f"  Downstream KPIs:  {', '.join(impact.get('downstream_kpis', []))}")
    print(f"  Cells at risk:    {len(impact.get('at_risk_cells', []))} cells")
    if impact.get("summary"):
        print(f"\n  LLM Summary:\n    {impact['summary']}")

    # ── Step 7: Root cause narrative ───────────────────────────────────
    print_section("Root Cause Analysis Narrative")
    rca = final_state.get("root_cause_summary", "")
    if rca:
        print(f"  {rca}")

    # ── Step 8: Generated test cases ──────────────────────────────────
    tests = final_state.get("generated_tests", [])
    print_section(f"Generated Test Cases ({len(tests)} total)")
    for i, test in enumerate(tests, 1):
        print(f"\n  [{i}] {test['name']}  ({test['priority']})")
        print(f"       ID: {test['id']}")
        if test.get("code"):
            # Print first 8 lines of code
            lines = test["code"].strip().split("\n")[:8]
            for line in lines:
                print(f"       {line}")
            if len(test["code"].strip().split("\n")) > 8:
                print(f"       ... ({len(test['code'].strip().split(chr(10)))} lines total)")

    # ── Step 9: Evaluation comparison ─────────────────────────────────
    if run_eval:
        run_evaluation(data, reasoner, event)

    print(f"\n{'═'*64}")
    print("  Demo complete.")
    print(f"{'═'*64}\n")

    return final_state


def run_evaluation(data: dict, reasoner: TeleKGReasoner, sample_event: dict):
    """
    Baseline vs. Proposed comparison for the paper's evaluation section.

    Baseline: Run ALL test cases (static test suite)
    Proposed: Run only KG-identified tests (dynamic selection)

    Metrics:
      - Test suite size reduction (%)
      - Coverage of KPI-impacted features (%)
      - P0/P1 critical test inclusion rate, RELATIVE TO THE TESTS THAT ARE
        ACTUALLY RELEVANT to this release (not the whole 18-test library —
        most of that library belongs to features this release never touches,
        so comparing against it understates the dynamic selection's recall)
    """
    print_section("Evaluation: Baseline vs. KG-Guided Dynamic Test Selection")

    all_tests = data["test_cases"]
    test_by_id = {t["id"]: t for t in all_tests}
    baseline_count = len(all_tests)

    # KG-guided selection
    release_id = sample_event["trigger"]
    impact = reasoner.release_impact(release_id)
    kg_tests = impact.get("required_tests", [])
    kg_count = len(kg_tests)

    # The reasoner's required_tests list only carries {id, name, priority},
    # not "feature" — join back against the full test catalog by ID to
    # recover which feature each selected test actually belongs to.
    kg_tests_full = [test_by_id.get(t["id"], t) for t in kg_tests]

    affected_features = set(impact.get("features", []))

    # Ground truth: every test in the FULL catalog that belongs to an
    # affected feature — this is what a perfect dynamic selector should
    # have picked. Comparing against the whole 18-test library (most of
    # which belongs to features this release never touches) would
    # understate recall.
    relevant_tests = [t for t in all_tests if t.get("feature") in affected_features]
    relevant_critical = [t for t in relevant_tests if t.get("priority") in ("P0", "P1")]

    kg_critical = [t for t in kg_tests_full if t.get("priority") in ("P0", "P1")]

    # Feature coverage: of the features this release touches, how many does
    # the selected test set actually have at least one test for?
    kg_features_covered = {t.get("feature") for t in kg_tests_full if t.get("feature")}
    feature_coverage = (
        len(kg_features_covered & affected_features) / max(len(affected_features), 1) * 100
        if affected_features else 0
    )

    reduction = (1 - kg_count / max(baseline_count, 1)) * 100

    # Critical inclusion rate = critical tests selected / critical tests
    # that are actually relevant to this release (not all 18 in the library)
    critical_inclusion = (
        len(kg_critical) / max(len(relevant_critical), 1) * 100
    ) if relevant_critical else 100.0

    print(f"\n  {'Metric':<44} {'Baseline':>10} {'KG-Guided':>10}")
    print(f"  {'─'*66}")
    print(f"  {'Total test cases executed':<44} {baseline_count:>10} {kg_count:>10}")
    print(f"  {'Test suite size reduction (%)':<44} {'—':>10} {reduction:>9.1f}%")
    print(f"  {'Relevant critical (P0/P1) tests for this release':<44} {len(relevant_critical):>10} {len(kg_critical):>10}")
    print(f"  {'Critical test inclusion rate (%) [relevant only]':<44} {'100%':>10} {critical_inclusion:>9.1f}%")
    print(f"  {'Affected feature coverage (%)':<44} {'—':>10} {feature_coverage:>9.1f}%")
    print(f"\n  ✓ KG-guided selection reduces test suite by {reduction:.1f}% "
          f"while including {critical_inclusion:.1f}% of the critical tests\n"
          f"    relevant to this release ({len(kg_critical)}/{len(relevant_critical)}), "
          f"and {feature_coverage:.0f}% feature coverage of touched features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TeleKG-Agent Demo")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock graph and LLM (no FalkorDB/Groq key needed)")
    parser.add_argument("--eval", action="store_true",
                        help="Run evaluation comparison after demo")
    args = parser.parse_args()
    run_demo(dry_run=args.dry_run, run_eval=args.eval)
