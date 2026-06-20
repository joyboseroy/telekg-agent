"""
Synthetic Telecom Network Generator

Creates a realistic simulated RAN environment:
  - 100 cells across 5 regions
  - 20 DUs, 5 CUs, 3 K8s clusters
  - All KPIs with realistic values
  - Test cases mapped to features
  - Software releases and their feature sets
  - Random change events that trigger the agentic pipeline

This replaces the need for a real OSS connection during prototyping.
"""

import random
import json
from datetime import datetime, timedelta
from telekg.schema import (
    CellNode, KPISnapshot, ChangeEvent,
    ALL_KPIS, FEATURE_KPI_IMPACTS, SW_RELEASE_FEATURES,
    KPI_CAUSAL_GRAPH, NodeType, EdgeType
)

random.seed(42)

REGIONS = ["North", "South", "East", "West", "Central"]
CLUSTERS = ["K8s_Cluster_A", "K8s_Cluster_B", "K8s_Cluster_C"]

# KPI baseline values and thresholds (realistic 5G RAN values)
KPI_BASELINES = {
    "PRB_Utilization":   {"mean": 55.0, "std": 15.0, "warn": 75.0, "crit": 90.0, "unit": "%"},
    "Throughput":        {"mean": 450.0,"std": 80.0, "warn": 200.0,"crit": 100.0,"unit": "Mbps"},
    "Latency":           {"mean": 12.0, "std": 3.0,  "warn": 20.0, "crit": 30.0, "unit": "ms"},
    "Accessibility":     {"mean": 98.5, "std": 1.5,  "warn": 95.0, "crit": 90.0, "unit": "%"},
    "Retainability":     {"mean": 99.2, "std": 0.8,  "warn": 97.0, "crit": 95.0, "unit": "%"},
    "Packet_Loss":       {"mean": 0.5,  "std": 0.3,  "warn": 1.5,  "crit": 3.0,  "unit": "%"},
    "Handover_Success":  {"mean": 97.0, "std": 2.0,  "warn": 94.0, "crit": 90.0, "unit": "%"},
    "Mobility":          {"mean": 96.5, "std": 2.5,  "warn": 93.0, "crit": 88.0, "unit": "%"},
    "User_Experience":   {"mean": 4.2,  "std": 0.4,  "warn": 3.5,  "crit": 3.0,  "unit": "MOS"},
    "Energy_Efficiency": {"mean": 72.0, "std": 10.0, "warn": 50.0, "crit": 35.0, "unit": "%"},
}

# Test case templates per feature
TEST_CASE_TEMPLATES = {
    "Energy_Saving": [
        {"id": "TC_ES_001", "name": "test_accessibility_during_sleep_mode",
         "validates": ["Accessibility"], "priority": "P1"},
        {"id": "TC_ES_002", "name": "test_prb_reduction_on_low_load",
         "validates": ["PRB_Utilization", "Energy_Efficiency"], "priority": "P2"},
        {"id": "TC_ES_003", "name": "test_wakeup_latency_on_traffic_burst",
         "validates": ["Latency", "Accessibility"], "priority": "P1"},
    ],
    "Cell_Sleep": [
        {"id": "TC_CS_001", "name": "test_sleep_entry_exit_kpis",
         "validates": ["Accessibility", "Latency"], "priority": "P1"},
        {"id": "TC_CS_002", "name": "test_neighbour_cell_load_on_sleep",
         "validates": ["PRB_Utilization", "Throughput"], "priority": "P2"},
    ],
    "Handover_Optimization": [
        {"id": "TC_HO_001", "name": "test_handover_success_rate_baseline",
         "validates": ["Handover_Success", "Mobility"], "priority": "P1"},
        {"id": "TC_HO_002", "name": "test_ping_pong_handover_rate",
         "validates": ["Mobility", "Retainability"], "priority": "P2"},
        {"id": "TC_HO_003", "name": "test_xn_handover_latency",
         "validates": ["Latency", "Handover_Success"], "priority": "P1"},
    ],
    "Load_Balancing": [
        {"id": "TC_LB_001", "name": "test_prb_redistribution_across_cells",
         "validates": ["PRB_Utilization", "Throughput"], "priority": "P1"},
        {"id": "TC_LB_002", "name": "test_ue_mobility_after_rebalance",
         "validates": ["User_Experience", "Mobility"], "priority": "P2"},
    ],
    "Beamforming": [
        {"id": "TC_BF_001", "name": "test_beam_tracking_throughput",
         "validates": ["Throughput", "Latency"], "priority": "P1"},
        {"id": "TC_BF_002", "name": "test_packet_loss_on_beam_switch",
         "validates": ["Packet_Loss", "Throughput"], "priority": "P1"},
    ],
    "URLLC_Slice": [
        {"id": "TC_UL_001", "name": "test_urllc_latency_sla",
         "validates": ["Latency", "Retainability"], "priority": "P0"},
        {"id": "TC_UL_002", "name": "test_urllc_packet_loss_under_load",
         "validates": ["Packet_Loss", "Reliability"], "priority": "P0"},
    ],
    "Massive_MIMO": [
        {"id": "TC_MM_001", "name": "test_mimo_throughput_gain",
         "validates": ["Throughput", "PRB_Utilization"], "priority": "P1"},
        {"id": "TC_MM_002", "name": "test_mimo_packet_loss_edge_ue",
         "validates": ["Packet_Loss", "Latency"], "priority": "P2"},
    ],
    "SON_Selfhealing": [
        {"id": "TC_SH_001", "name": "test_selfhealing_accessibility_recovery",
         "validates": ["Accessibility", "Retainability"], "priority": "P1"},
        {"id": "TC_SH_002", "name": "test_coverage_hole_detection",
         "validates": ["Handover_Success", "Accessibility"], "priority": "P2"},
    ],
}


def generate_cells(n=100) -> list[CellNode]:
    cells = []
    features = list(FEATURE_KPI_IMPACTS.keys())
    for i in range(n):
        region = REGIONS[i % len(REGIONS)]
        du_id = f"DU_{(i // 5) + 1:02d}"
        cluster = CLUSTERS[i % len(CLUSTERS)]
        # Each cell uses 2-4 features
        active = random.sample(features, k=random.randint(2, 4))
        cells.append(CellNode(
            cell_id=f"Cell_{i+1:03d}",
            region=region,
            du_id=du_id,
            cluster=cluster,
            active_features=active,
            lat=round(12.9 + random.uniform(-1.5, 1.5), 4),
            lon=round(77.6 + random.uniform(-1.5, 1.5), 4),
        ))
    return cells


def generate_kpi_snapshot(cell_id: str, timestamp: str, degraded_kpis=None) -> list[dict]:
    """Generate KPI values for a cell; optionally degrade specific KPIs to simulate impact."""
    snapshots = []
    degraded_kpis = degraded_kpis or []
    for kpi, spec in KPI_BASELINES.items():
        value = random.gauss(spec["mean"], spec["std"])
        if kpi in degraded_kpis:
            # Push toward critical threshold
            if kpi in ["Throughput", "Accessibility", "Retainability", "Handover_Success",
                       "Mobility", "User_Experience", "Energy_Efficiency"]:
                value = random.gauss(spec["crit"] * 1.05, spec["std"] * 0.5)
            else:
                value = random.gauss(spec["crit"] * 0.95, spec["std"] * 0.5)
        snapshots.append({
            "kpi": kpi,
            "cell_id": cell_id,
            "value": round(max(0, value), 3),
            "warn": spec["warn"],
            "crit": spec["crit"],
            "unit": spec["unit"],
            "timestamp": timestamp,
        })
    return snapshots


def generate_change_events(cells: list[CellNode], n_events=20) -> list[dict]:
    """Generate a stream of network change events."""
    events = []
    base_time = datetime(2025, 1, 1, 8, 0, 0)
    releases = list(SW_RELEASE_FEATURES.keys())
    cell_ids = [c.cell_id for c in cells]

    event_types = [
        ("SW_UPGRADE",    "Software upgrade to {trigger} deployed"),
        ("CONFIG_CHANGE", "Configuration parameter change: {trigger}"),
        ("CELL_OUTAGE",   "Unplanned outage on cells: {trigger}"),
        ("SLICE_CREATE",  "New network slice provisioned: {trigger}"),
    ]

    for i in range(n_events):
        etype, desc_template = random.choice(event_types)
        timestamp = (base_time + timedelta(hours=i * 12 + random.randint(0, 8))).isoformat()
        n_affected = random.randint(3, 15)
        affected = random.sample(cell_ids, k=n_affected)

        if etype == "SW_UPGRADE":
            trigger = random.choice(releases)
        elif etype == "CONFIG_CHANGE":
            trigger = f"energy_saving_threshold={random.randint(20, 80)}"
        elif etype == "CELL_OUTAGE":
            trigger = ", ".join(random.sample(cell_ids, 2))
        else:
            trigger = f"Slice_URLLC_{i}"

        events.append({
            "event_id": f"EVT_{i+1:04d}",
            "event_type": etype,
            "trigger": trigger,
            "affected_cells": affected,
            "timestamp": timestamp,
            "description": desc_template.format(trigger=trigger),
        })
    return events


def build_full_dataset() -> dict:
    """Build and return the complete synthetic network dataset."""
    cells = generate_cells(100)
    ts_now = datetime.now().isoformat()
    all_kpis = []
    for cell in cells:
        all_kpis.extend(generate_kpi_snapshot(cell.cell_id, ts_now))

    # Generate before/after KPI snapshots for 5 change events
    change_events = generate_change_events(cells, n_events=20)

    # Flatten test cases
    all_tests = []
    for feature, tcs in TEST_CASE_TEMPLATES.items():
        for tc in tcs:
            all_tests.append({**tc, "feature": feature})

    return {
        "cells": [c.__dict__ for c in cells],
        "kpis": all_kpis,
        "change_events": change_events,
        "test_cases": all_tests,
        "kpi_causal_graph": [
            {"from": f, "to": t, "weight": w} for f, t, w in KPI_CAUSAL_GRAPH
        ],
        "feature_kpi_impacts": FEATURE_KPI_IMPACTS,
        "sw_release_features": SW_RELEASE_FEATURES,
    }


if __name__ == "__main__":
    import json, pathlib
    data = build_full_dataset()
    out = pathlib.Path("data/synthetic_network.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    print(f"Generated synthetic network:")
    print(f"  Cells: {len(data['cells'])}")
    print(f"  KPI snapshots: {len(data['kpis'])}")
    print(f"  Change events: {len(data['change_events'])}")
    print(f"  Test cases: {len(data['test_cases'])}")
    print(f"  Written to {out}")
