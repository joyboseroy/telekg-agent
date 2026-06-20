"""
demo_telecom_graph.py

A standalone, zero-dependency demo of the causal reasoning core that powers
TeleKG-Agent's ImpactAnalysisAgent and RootCauseAgent (see telekg/reasoner.py
and agents/pipeline.py for the full version with FalkorDB + LangGraph + Groq).

This file exists for one purpose: a 30-second, no-setup demo. Drop it in the
repo root and run it with nothing but the Python standard library. No
FalkorDB, no API key, no pip install. It's a deliberately small in-memory
version of the same KPI causality graph (telekg/schema.py's L2 layer),
intended for live demos and coding-round walkthroughs where you don't have
time to spin up infrastructure.

Usage:
    python demo_telecom_graph.py
    python demo_telecom_graph.py --kpi throughput
    python demo_telecom_graph.py --list

For the real system (real FalkorDB graph, 5-agent LangGraph pipeline, Groq
narration, full PM counter substrate), see main.py and the rest of this repo.
"""

import argparse
from collections import defaultdict, deque


# --- Minimal causal graph ------------------------------------------------
# Same conceptual shape as telekg/schema.py's L2 KPI-causality layer
# (KPI --[IMPACTS, weight]--> KPI), but hand-written and in-memory here
# instead of loaded from FalkorDB, so this file has zero dependencies.
#
# Each edge: (cause, effect, weight, evidence)
# weight is a rough causal-strength estimate (0-1), evidence is a short
# plain-language justification, standing in for the 3GPP/ETSI references
# used in the real pm_registry.py.

CAUSAL_EDGES = [
    ("prb_utilization_high", "throughput", 0.85, "PRB congestion limits scheduled resource blocks per UE"),
    ("interference_high", "throughput", 0.55, "Higher interference reduces effective SINR and MCS selection"),
    ("handover_failure_rate_high", "throughput", 0.40, "Failed handovers cause re-establishment delay and retransmission"),
    ("rrc_setup_failure_high", "accessibility", 0.90, "RRC setup is the first step of call/session establishment"),
    ("paging_failure_high", "accessibility", 0.45, "Failed paging prevents the UE from receiving the setup request"),
    ("backhaul_congestion", "latency", 0.75, "Backhaul congestion queues packets before they reach the core"),
    ("prb_utilization_high", "latency", 0.50, "Scheduler queuing delay increases under high PRB load"),
    ("cell_outage", "accessibility", 0.95, "No accessible cell means no successful setup attempts"),
    ("cell_outage", "throughput", 0.95, "No serving cell means zero throughput for affected UEs"),
    ("energy_saving_feature_active", "prb_utilization_high", 0.30, "Reduced active carriers can concentrate load onto fewer PRBs"),
    ("software_release_v2_3", "rrc_setup_failure_high", 0.20, "Known regression in RRC state machine in this release (synthetic example)"),
]

# Remedies, keyed by cause
REMEDIES = {
    "prb_utilization_high": "Add carrier capacity or enable load balancing across neighboring cells",
    "interference_high": "Run interference mitigation (ICIC) or check for rogue/external sources",
    "handover_failure_rate_high": "Review handover thresholds and neighbor relations",
    "rrc_setup_failure_high": "Check RRC config and core network signaling path",
    "paging_failure_high": "Verify paging channel capacity and core paging timers",
    "backhaul_congestion": "Increase backhaul capacity or apply QoS shaping",
    "cell_outage": "Dispatch field team / restart affected DU-CU pair",
    "energy_saving_feature_active": "Tune energy-saving thresholds to avoid concentrating load",
    "software_release_v2_3": "Apply hotfix or roll back to previous release",
}


def build_reverse_graph():
    """effect -> list of (cause, weight, evidence)"""
    reverse = defaultdict(list)
    for cause, effect, weight, evidence in CAUSAL_EDGES:
        reverse[effect].append((cause, weight, evidence))
    return reverse


def build_forward_graph():
    """cause -> list of (effect, weight, evidence), for tracing multi-hop chains"""
    forward = defaultdict(list)
    for cause, effect, weight, evidence in CAUSAL_EDGES:
        forward[cause].append((effect, weight, evidence))
    return forward


def explain(kpi: str, max_depth: int = 4):
    """
    Pure graph traversal: given a degraded KPI, walk backward through the
    causal graph and return every root cause, with the full multi-hop path
    and a recommended remedy. No LLM involved, this is the same traversal
    logic as telekg/reasoner.py's root_cause(), just without the FalkorDB
    backend or the narration layer on top.
    """
    reverse = build_reverse_graph()
    paths = []

    # BFS backward from the KPI, collecting all causal paths
    queue = deque([(kpi, [], 0)])
    while queue:
        node, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        causes = reverse.get(node, [])
        if not causes:
            continue
        for cause, weight, evidence in causes:
            new_path = path + [(cause, node, weight, evidence)]
            paths.append(new_path)
            queue.append((cause, new_path, depth + 1))

    return paths


def print_explanation(kpi: str):
    paths = explain(kpi)
    if not paths:
        print(f"No causal edges found for KPI '{kpi}'. Try --list to see available KPIs.")
        return

    print(f"\nWhy is '{kpi}' degraded? Causal chains found (deepest first):\n")
    # sort by path length descending, then by weight of the root cause
    paths_sorted = sorted(paths, key=lambda p: (-len(p), -p[-1][2]))

    seen_roots = set()
    for path in paths_sorted:
        root_cause = path[-1][0]
        if root_cause in seen_roots:
            continue
        seen_roots.add(root_cause)

        chain_str = " -> ".join([path[-1][0]] + [step[1] for step in reversed(path)])
        print(f"  {chain_str}")
        for step in reversed(path):
            cause, effect, weight, evidence = step
            print(f"      {cause} --[weight={weight}]--> {effect}")
            print(f"          evidence: {evidence}")
        remedy = REMEDIES.get(root_cause, "No remedy on file for this cause")
        print(f"      remedy: {remedy}\n")


def list_kpis():
    reverse = build_reverse_graph()
    print("\nKPIs with known causal chains in this demo graph:")
    for kpi in sorted(reverse.keys()):
        n_causes = len(reverse[kpi])
        print(f"  {kpi:35s} ({n_causes} direct cause{'s' if n_causes != 1 else ''})")
    print("\nRun with --kpi <name> to see the full causal explanation.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-dependency demo of TeleKG-Agent's causal graph reasoning. "
                     "See main.py for the full system (FalkorDB + 5-agent LangGraph pipeline + Groq)."
    )
    parser.add_argument("--kpi", default="throughput", help="KPI to explain (default: throughput)")
    parser.add_argument("--list", action="store_true", help="List all KPIs with known causal chains")
    args = parser.parse_args()

    if args.list:
        list_kpis()
    else:
        print_explanation(args.kpi)


if __name__ == "__main__":
    main()
