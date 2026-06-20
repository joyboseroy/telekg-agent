"""
TeleKG Schema: Node types, edge types, and ontology for the Telecom Knowledge Graph.

The schema encodes three layers:
  L1 - Physical/logical network topology (cells, DUs, CUs, clusters)
  L2 - KPI dependency graph (causal relationships between KPIs)
  L3 - Feature/software/test graph (what features affect what KPIs, what tests cover them)

All three layers are unified in a single FalkorDB graph, enabling cross-layer
traversal: "software release X affects feature Y which degrades KPI Z in cells A,B,C —
generate tests T1,T2,T3 to validate the regression."
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────
# Node Types
# ─────────────────────────────────────────────

class NodeType(str, Enum):
    # L1: Network topology
    CELL        = "Cell"
    DU          = "DistributedUnit"
    CU          = "CentralUnit"
    K8S_CLUSTER = "K8sCluster"
    REGION      = "Region"
    SLICE       = "NetworkSlice"

    # L2: KPI layer
    KPI         = "KPI"
    ALARM       = "Alarm"

    # L3: Software / feature / test layer
    FEATURE     = "Feature"
    SW_RELEASE  = "SoftwareRelease"
    CONFIG      = "Configuration"
    TEST_CASE   = "TestCase"

    # Events (temporal layer)
    CHANGE_EVENT = "ChangeEvent"


# ─────────────────────────────────────────────
# Edge Types
# ─────────────────────────────────────────────

class EdgeType(str, Enum):
    # Topology edges
    SERVED_BY       = "SERVED_BY"        # Cell → DU
    HOSTED_ON       = "HOSTED_ON"        # DU → K8sCluster
    PART_OF_REGION  = "PART_OF_REGION"   # Cell → Region
    PART_OF_SLICE   = "PART_OF_SLICE"    # Cell → NetworkSlice

    # KPI causal edges (L2)
    IMPACTS         = "IMPACTS"          # KPI → KPI  (causal)
    HAS_KPI         = "HAS_KPI"          # Cell → KPI
    HAS_ALARM       = "HAS_ALARM"        # Cell/DU → Alarm
    TRIGGERS_ALARM  = "TRIGGERS_ALARM"   # KPI → Alarm

    # Feature/software edges (L3)
    AFFECTS_FEATURE = "AFFECTS_FEATURE"  # SoftwareRelease → Feature
    AFFECTS_KPI     = "AFFECTS_KPI"      # Feature → KPI
    USES_FEATURE    = "USES_FEATURE"     # Cell → Feature
    REQUIRES_CONFIG = "REQUIRES_CONFIG"  # Feature → Configuration
    COVERED_BY      = "COVERED_BY"       # Feature → TestCase
    VALIDATES       = "VALIDATES"        # TestCase → KPI

    # Temporal / event edges
    CAUSED_BY       = "CAUSED_BY"        # ChangeEvent → (SWRelease | Config)
    AFFECTS_CELL    = "AFFECTS_CELL"     # ChangeEvent → Cell
    BEFORE_STATE    = "BEFORE_STATE"     # ChangeEvent → KPI (snapshot)
    AFTER_STATE     = "AFTER_STATE"      # ChangeEvent → KPI (snapshot)


# ─────────────────────────────────────────────
# KPI Ontology — encodes telecom domain expertise
# ─────────────────────────────────────────────

# Each tuple: (source_kpi, target_kpi, impact_weight 0..1)
KPI_CAUSAL_GRAPH = [
    ("PRB_Utilization",     "Throughput",           0.85),
    ("PRB_Utilization",     "Latency",              0.70),
    ("Latency",             "Accessibility",        0.75),
    ("Packet_Loss",         "Retainability",        0.90),
    ("Packet_Loss",         "Throughput",           0.60),
    ("Throughput",          "User_Experience",      0.80),
    ("Retainability",       "User_Experience",      0.75),
    ("Accessibility",       "User_Experience",      0.70),
    ("Handover_Success",    "Mobility",             0.95),
    ("Mobility",            "Retainability",        0.65),
    ("Energy_Efficiency",   "PRB_Utilization",      0.50),
]

ALL_KPIS = list({k for pair in KPI_CAUSAL_GRAPH for k in pair[:2]})

# Feature → KPI impact mapping (which KPIs a feature can degrade/improve)
FEATURE_KPI_IMPACTS = {
    "Energy_Saving":        ["PRB_Utilization", "Accessibility", "Throughput"],
    "Cell_Sleep":           ["Accessibility", "Latency", "Energy_Efficiency"],
    "Handover_Optimization":["Handover_Success", "Mobility", "Latency"],
    "Load_Balancing":       ["PRB_Utilization", "Throughput", "User_Experience"],
    "Beamforming":          ["Throughput", "Latency", "Packet_Loss"],
    "URLLC_Slice":          ["Latency", "Retainability", "Packet_Loss"],
    "Massive_MIMO":         ["Throughput", "PRB_Utilization", "Packet_Loss"],
    "SON_Selfhealing":      ["Accessibility", "Retainability", "Handover_Success"],
}

# Software releases and which features they touch
SW_RELEASE_FEATURES = {
    "Release_24.1": ["Energy_Saving", "Cell_Sleep"],
    "Release_24.2": ["Handover_Optimization", "Load_Balancing"],
    "Release_25.1": ["Beamforming", "Massive_MIMO", "URLLC_Slice"],
    "Release_25.2": ["Energy_Saving", "SON_Selfhealing", "Cell_Sleep"],
}


# ─────────────────────────────────────────────
# Dataclasses for typed node/edge construction
# ─────────────────────────────────────────────

@dataclass
class CellNode:
    cell_id: str
    region: str
    du_id: str
    cluster: str
    active_features: list[str] = field(default_factory=list)
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class KPISnapshot:
    kpi_name: str
    cell_id: str
    value: float
    threshold_warn: float
    threshold_crit: float
    timestamp: str


@dataclass
class ChangeEvent:
    event_id: str
    event_type: str          # "SW_UPGRADE" | "CONFIG_CHANGE" | "CELL_OUTAGE" | "SLICE_CREATE"
    trigger: str             # e.g. "Release_25.2"
    affected_cells: list[str]
    timestamp: str
    description: str
