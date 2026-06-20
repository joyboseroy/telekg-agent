"""
PM Counter Registry and KPI Formula Definitions

This module is the missing layer between raw telemetry and named KPIs.
In a real OSS, KPIs are NOT primitive measurements — they are formulas
computed over Performance Monitoring (PM) counters, which are the actual
primitive measurements a base station reports (e.g. RRC_SETUP_SUCCESS,
RRC_SETUP_ATTEMPT).

This registry encodes:
  1. The PM counter dictionary  — name, unit, description, typical range
  2. The KPI formula registry   — each KPI expressed as a formula over counters
  3. A small synonym/alias table — the vocabulary a user might use informally
     when typing a custom KPI definition (used as weak supervision / retrieval
     anchors for the NL -> formula translator, NOT a lookup shortcut — the
     translator is still expected to handle paraphrases not in this table)

This is the ground truth against which the LLM-based translator (translator.py)
is evaluated: given a free-text KPI definition, does it retrieve the right
counters and reconstruct the correct formula?
"""

from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────
# PM Counter Dictionary
# ─────────────────────────────────────────────

@dataclass
class PMCounter:
    name: str
    description: str
    unit: str
    counter_type: str   # "cumulative" | "gauge" | "duration"


PM_COUNTERS: dict[str, PMCounter] = {
    "RRC_SETUP_ATTEMPT":        PMCounter("RRC_SETUP_ATTEMPT", "Number of RRC connection setup attempts", "count", "cumulative"),
    "RRC_SETUP_SUCCESS":        PMCounter("RRC_SETUP_SUCCESS", "Number of successful RRC connection setups", "count", "cumulative"),
    "ERAB_SETUP_ATTEMPT":       PMCounter("ERAB_SETUP_ATTEMPT", "Number of E-RAB setup attempts", "count", "cumulative"),
    "ERAB_SETUP_SUCCESS":       PMCounter("ERAB_SETUP_SUCCESS", "Number of successful E-RAB setups", "count", "cumulative"),
    "ERAB_DROP_COUNT":          PMCounter("ERAB_DROP_COUNT", "Number of abnormally released E-RABs", "count", "cumulative"),
    "ERAB_NORMAL_RELEASE":      PMCounter("ERAB_NORMAL_RELEASE", "Number of normally released E-RABs", "count", "cumulative"),
    "HO_ATTEMPT_OUT":           PMCounter("HO_ATTEMPT_OUT", "Outgoing handover attempts", "count", "cumulative"),
    "HO_SUCCESS_OUT":           PMCounter("HO_SUCCESS_OUT", "Successful outgoing handovers", "count", "cumulative"),
    "HO_ATTEMPT_IN":            PMCounter("HO_ATTEMPT_IN", "Incoming handover attempts", "count", "cumulative"),
    "HO_SUCCESS_IN":            PMCounter("HO_SUCCESS_IN", "Successful incoming handovers", "count", "cumulative"),
    "PRB_USED_DL":              PMCounter("PRB_USED_DL", "Physical Resource Blocks used, downlink", "count", "gauge"),
    "PRB_AVAILABLE_DL":         PMCounter("PRB_AVAILABLE_DL", "Physical Resource Blocks available, downlink", "count", "gauge"),
    "PRB_USED_UL":              PMCounter("PRB_USED_UL", "Physical Resource Blocks used, uplink", "count", "gauge"),
    "PRB_AVAILABLE_UL":         PMCounter("PRB_AVAILABLE_UL", "Physical Resource Blocks available, uplink", "count", "gauge"),
    "DL_VOL_BYTES":             PMCounter("DL_VOL_BYTES", "Downlink data volume transferred", "bytes", "cumulative"),
    "UL_VOL_BYTES":             PMCounter("UL_VOL_BYTES", "Uplink data volume transferred", "bytes", "cumulative"),
    "DL_SCHED_TIME_MS":         PMCounter("DL_SCHED_TIME_MS", "Total downlink scheduled time", "ms", "cumulative"),
    "UL_SCHED_TIME_MS":         PMCounter("UL_SCHED_TIME_MS", "Total uplink scheduled time", "ms", "cumulative"),
    "PDCP_PACKETS_SENT_DL":     PMCounter("PDCP_PACKETS_SENT_DL", "PDCP layer packets sent, downlink", "count", "cumulative"),
    "PDCP_PACKETS_LOST_DL":     PMCounter("PDCP_PACKETS_LOST_DL", "PDCP layer packets lost, downlink", "count", "cumulative"),
    "PDCP_PACKETS_SENT_UL":     PMCounter("PDCP_PACKETS_SENT_UL", "PDCP layer packets sent, uplink", "count", "cumulative"),
    "PDCP_PACKETS_LOST_UL":     PMCounter("PDCP_PACKETS_LOST_UL", "PDCP layer packets lost, uplink", "count", "cumulative"),
    "UE_CONTEXT_RELEASE_RADIO": PMCounter("UE_CONTEXT_RELEASE_RADIO", "UE context releases due to radio link failure", "count", "cumulative"),
    "RRC_CONN_MEAN_DURATION_S": PMCounter("RRC_CONN_MEAN_DURATION_S", "Mean RRC connection duration", "seconds", "duration"),
    "CELL_AVAILABLE_TIME_S":    PMCounter("CELL_AVAILABLE_TIME_S", "Time the cell was available for service", "seconds", "duration"),
    "CELL_TOTAL_TIME_S":        PMCounter("CELL_TOTAL_TIME_S", "Total observation period for the cell", "seconds", "duration"),
    "RTT_SUM_MS":               PMCounter("RTT_SUM_MS", "Sum of round-trip-time samples", "ms", "cumulative"),
    "RTT_SAMPLE_COUNT":         PMCounter("RTT_SAMPLE_COUNT", "Number of RTT samples taken", "count", "cumulative"),
    "RRU_POWER_CONSUMPTION_W":  PMCounter("RRU_POWER_CONSUMPTION_W", "RRU power consumption", "watts", "gauge"),
    "RRU_RATED_POWER_W":        PMCounter("RRU_RATED_POWER_W", "RRU rated maximum power", "watts", "gauge"),
}


# ─────────────────────────────────────────────
# KPI Formula Registry
# ─────────────────────────────────────────────

@dataclass
class KPIFormula:
    kpi_name: str
    formula_str: str                 # human-readable formula, e.g. "a / b * 100"
    counters_used: list[str]
    compute: Callable[[dict], float] # function(counter_values: dict) -> float
    unit: str
    standard_ref: str = ""           # e.g. 3GPP TS 28.554 clause


def _safe_div(num, denom, default=0.0):
    return (num / denom) if denom else default


KPI_REGISTRY: dict[str, KPIFormula] = {

    "Accessibility": KPIFormula(
        kpi_name="Accessibility",
        formula_str="RRC_SETUP_SUCCESS / RRC_SETUP_ATTEMPT * 100",
        counters_used=["RRC_SETUP_SUCCESS", "RRC_SETUP_ATTEMPT"],
        compute=lambda c: _safe_div(c["RRC_SETUP_SUCCESS"], c["RRC_SETUP_ATTEMPT"]) * 100,
        unit="%",
        standard_ref="3GPP TS 28.554 5.1.1",
    ),

    "Retainability": KPIFormula(
        kpi_name="Retainability",
        formula_str="(1 - ERAB_DROP_COUNT / (ERAB_DROP_COUNT + ERAB_NORMAL_RELEASE)) * 100",
        counters_used=["ERAB_DROP_COUNT", "ERAB_NORMAL_RELEASE"],
        compute=lambda c: (1 - _safe_div(c["ERAB_DROP_COUNT"],
                                          c["ERAB_DROP_COUNT"] + c["ERAB_NORMAL_RELEASE"], default=0)) * 100,
        unit="%",
        standard_ref="3GPP TS 28.554 5.1.2",
    ),

    "Handover_Success": KPIFormula(
        kpi_name="Handover_Success",
        formula_str="(HO_SUCCESS_OUT + HO_SUCCESS_IN) / (HO_ATTEMPT_OUT + HO_ATTEMPT_IN) * 100",
        counters_used=["HO_SUCCESS_OUT", "HO_SUCCESS_IN", "HO_ATTEMPT_OUT", "HO_ATTEMPT_IN"],
        compute=lambda c: _safe_div(
            c["HO_SUCCESS_OUT"] + c["HO_SUCCESS_IN"],
            c["HO_ATTEMPT_OUT"] + c["HO_ATTEMPT_IN"]
        ) * 100,
        unit="%",
        standard_ref="3GPP TS 28.554 5.1.3",
    ),

    "PRB_Utilization": KPIFormula(
        kpi_name="PRB_Utilization",
        formula_str="PRB_USED_DL / PRB_AVAILABLE_DL * 100",
        counters_used=["PRB_USED_DL", "PRB_AVAILABLE_DL"],
        compute=lambda c: _safe_div(c["PRB_USED_DL"], c["PRB_AVAILABLE_DL"]) * 100,
        unit="%",
        standard_ref="3GPP TS 28.552 5.1.1.1",
    ),

    "Throughput": KPIFormula(
        kpi_name="Throughput",
        formula_str="(DL_VOL_BYTES * 8) / (DL_SCHED_TIME_MS / 1000) / 1e6",
        counters_used=["DL_VOL_BYTES", "DL_SCHED_TIME_MS"],
        compute=lambda c: _safe_div(c["DL_VOL_BYTES"] * 8, c["DL_SCHED_TIME_MS"] / 1000) / 1e6,
        unit="Mbps",
        standard_ref="3GPP TS 28.552 5.1.3.1",
    ),

    "Packet_Loss": KPIFormula(
        kpi_name="Packet_Loss",
        formula_str="PDCP_PACKETS_LOST_DL / PDCP_PACKETS_SENT_DL * 100",
        counters_used=["PDCP_PACKETS_LOST_DL", "PDCP_PACKETS_SENT_DL"],
        compute=lambda c: _safe_div(c["PDCP_PACKETS_LOST_DL"], c["PDCP_PACKETS_SENT_DL"]) * 100,
        unit="%",
        standard_ref="3GPP TS 28.552 5.1.4.2",
    ),

    "Latency": KPIFormula(
        kpi_name="Latency",
        formula_str="RTT_SUM_MS / RTT_SAMPLE_COUNT",
        counters_used=["RTT_SUM_MS", "RTT_SAMPLE_COUNT"],
        compute=lambda c: _safe_div(c["RTT_SUM_MS"], c["RTT_SAMPLE_COUNT"]),
        unit="ms",
        standard_ref="3GPP TS 28.552 5.1.5.1",
    ),

    "Mobility": KPIFormula(
        kpi_name="Mobility",
        formula_str="1 - (UE_CONTEXT_RELEASE_RADIO / (HO_ATTEMPT_OUT + 1))",
        counters_used=["UE_CONTEXT_RELEASE_RADIO", "HO_ATTEMPT_OUT"],
        compute=lambda c: (1 - _safe_div(c["UE_CONTEXT_RELEASE_RADIO"], c["HO_ATTEMPT_OUT"] + 1)) * 100,
        unit="%",
        standard_ref="Derived (composite)",
    ),

    "Cell_Availability": KPIFormula(
        kpi_name="Cell_Availability",
        formula_str="CELL_AVAILABLE_TIME_S / CELL_TOTAL_TIME_S * 100",
        counters_used=["CELL_AVAILABLE_TIME_S", "CELL_TOTAL_TIME_S"],
        compute=lambda c: _safe_div(c["CELL_AVAILABLE_TIME_S"], c["CELL_TOTAL_TIME_S"]) * 100,
        unit="%",
        standard_ref="3GPP TS 28.552 5.1.2.1",
    ),

    "Energy_Efficiency": KPIFormula(
        kpi_name="Energy_Efficiency",
        formula_str="(1 - RRU_POWER_CONSUMPTION_W / RRU_RATED_POWER_W) * 100",
        counters_used=["RRU_POWER_CONSUMPTION_W", "RRU_RATED_POWER_W"],
        compute=lambda c: (1 - _safe_div(c["RRU_POWER_CONSUMPTION_W"], c["RRU_RATED_POWER_W"])) * 100,
        unit="%",
        standard_ref="ETSI ES 203 228",
    ),
}


# ─────────────────────────────────────────────
# Informal vocabulary table
# (weak anchors for retrieval; NOT a shortcut lookup table)
# ─────────────────────────────────────────────

INFORMAL_ALIASES: dict[str, str] = {
    "call drop rate": "Retainability",
    "drop rate": "Retainability",
    "call success rate": "Accessibility",
    "connection success rate": "Accessibility",
    "setup success rate": "Accessibility",
    "handover success rate": "Handover_Success",
    "ho success": "Handover_Success",
    "cell load": "PRB_Utilization",
    "resource utilization": "PRB_Utilization",
    "data speed": "Throughput",
    "download speed": "Throughput",
    "packet loss rate": "Packet_Loss",
    "round trip delay": "Latency",
    "ping time": "Latency",
    "uptime": "Cell_Availability",
    "power efficiency": "Energy_Efficiency",
}


def get_counter_names() -> list[str]:
    return list(PM_COUNTERS.keys())


def get_kpi_names() -> list[str]:
    return list(KPI_REGISTRY.keys())


def compute_kpi(kpi_name: str, counter_values: dict[str, float]) -> float:
    """Compute a registered KPI's value given a dict of counter values."""
    if kpi_name not in KPI_REGISTRY:
        raise KeyError(f"Unknown KPI: {kpi_name}")
    formula = KPI_REGISTRY[kpi_name]
    missing = [c for c in formula.counters_used if c not in counter_values]
    if missing:
        raise ValueError(f"Missing counters for {kpi_name}: {missing}")
    return formula.compute(counter_values)
