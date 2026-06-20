"""
PM Counter Telemetry Generator

Generates realistic raw PM counter values per cell, from which KPIs in
pm_registry.py are DERIVED (not sampled directly). This is the layer
underneath telekg/simulator.py's KPI snapshots — it answers "where did
that Accessibility number actually come from?"

Counter values are generated so that the derived KPI lands close to the
KPI_BASELINES used elsewhere in the codebase, preserving consistency
between the two simulators while making the derivation auditable.
"""

import random
from telekg.pm_registry import KPI_REGISTRY, compute_kpi

random.seed(7)


def _gen_accessibility_counters(target_pct: float, attempts: int = 5000) -> dict:
    success = int(attempts * target_pct / 100)
    return {"RRC_SETUP_ATTEMPT": attempts, "RRC_SETUP_SUCCESS": success}


def _gen_retainability_counters(target_pct: float, total_erabs: int = 8000) -> dict:
    drop_frac = 1 - target_pct / 100
    drops = int(total_erabs * drop_frac)
    normal = total_erabs - drops
    return {"ERAB_DROP_COUNT": drops, "ERAB_NORMAL_RELEASE": normal}


def _gen_handover_counters(target_pct: float, attempts_out: int = 1200, attempts_in: int = 1100) -> dict:
    total_attempts = attempts_out + attempts_in
    total_success = int(total_attempts * target_pct / 100)
    success_out = int(total_success * (attempts_out / total_attempts))
    success_in = total_success - success_out
    return {
        "HO_ATTEMPT_OUT": attempts_out, "HO_SUCCESS_OUT": success_out,
        "HO_ATTEMPT_IN": attempts_in, "HO_SUCCESS_IN": success_in,
    }


def _gen_prb_counters(target_pct: float, available: int = 1000) -> dict:
    used = int(available * target_pct / 100)
    return {"PRB_USED_DL": used, "PRB_AVAILABLE_DL": available}


def _gen_throughput_counters(target_mbps: float, sched_ms: int = 900_000) -> dict:
    bytes_needed = int((target_mbps * 1e6 * (sched_ms / 1000)) / 8)
    return {"DL_VOL_BYTES": bytes_needed, "DL_SCHED_TIME_MS": sched_ms}


def _gen_packet_loss_counters(target_pct: float, sent: int = 2_000_000) -> dict:
    lost = int(sent * target_pct / 100)
    return {"PDCP_PACKETS_LOST_DL": lost, "PDCP_PACKETS_SENT_DL": sent}


def _gen_latency_counters(target_ms: float, samples: int = 10_000) -> dict:
    rtt_sum = int(target_ms * samples)
    return {"RTT_SUM_MS": rtt_sum, "RTT_SAMPLE_COUNT": samples}


def _gen_mobility_counters(target_pct: float, ho_attempts: int = 1200) -> dict:
    release_frac = 1 - target_pct / 100
    releases = int(ho_attempts * release_frac)
    return {"UE_CONTEXT_RELEASE_RADIO": releases, "HO_ATTEMPT_OUT": ho_attempts}


def _gen_availability_counters(target_pct: float, total_s: int = 86400) -> dict:
    avail = int(total_s * target_pct / 100)
    return {"CELL_AVAILABLE_TIME_S": avail, "CELL_TOTAL_TIME_S": total_s}


def _gen_energy_counters(target_pct: float, rated_w: float = 1200.0) -> dict:
    consumption = rated_w * (1 - target_pct / 100)
    return {"RRU_POWER_CONSUMPTION_W": round(consumption, 1), "RRU_RATED_POWER_W": rated_w}


# Map KPI name -> (generator_fn, target_kwarg_name)
_KPI_TO_GENERATOR = {
    "Accessibility":     _gen_accessibility_counters,
    "Retainability":      _gen_retainability_counters,
    "Handover_Success":   _gen_handover_counters,
    "PRB_Utilization":    _gen_prb_counters,
    "Throughput":         _gen_throughput_counters,
    "Packet_Loss":        _gen_packet_loss_counters,
    "Latency":            _gen_latency_counters,
    "Mobility":           _gen_mobility_counters,
    "Cell_Availability":  _gen_availability_counters,
    "Energy_Efficiency":  _gen_energy_counters,
}

# Realistic target value distributions (mean, std) per KPI, used to drive
# the counter generators above so the DERIVED kpi value looks realistic.
_KPI_TARGET_DIST = {
    "Accessibility":    (98.5, 1.2),
    "Retainability":     (99.2, 0.6),
    "Handover_Success":  (97.0, 1.8),
    "PRB_Utilization":   (55.0, 14.0),
    "Throughput":        (450.0, 70.0),
    "Packet_Loss":       (0.5, 0.3),
    "Latency":           (12.0, 3.0),
    "Mobility":          (96.5, 2.2),
    "Cell_Availability": (99.6, 0.5),
    "Energy_Efficiency": (72.0, 9.0),
}


def generate_cell_counters(cell_id: str, degraded_kpis: list[str] = None) -> dict:
    """
    Generate one full set of raw PM counters for a cell, such that every
    registered KPI is computable from these counters via pm_registry.compute_kpi.

    Returns: {"cell_id": ..., "counters": {counter_name: value, ...}}
    """
    degraded_kpis = degraded_kpis or []
    all_counters = {}

    for kpi_name, gen_fn in _KPI_TO_GENERATOR.items():
        mean, std = _KPI_TARGET_DIST[kpi_name]
        target = random.gauss(mean, std)

        if kpi_name in degraded_kpis:
            # Push target toward a degraded value (direction depends on KPI polarity)
            if kpi_name in ("PRB_Utilization", "Packet_Loss", "Latency"):
                target = mean + std * random.uniform(2.5, 4.0)   # higher = worse
            else:
                target = mean - std * random.uniform(2.5, 4.0)   # lower = worse

        target = max(0.01, target)
        counters = gen_fn(target)
        all_counters.update(counters)

    return {"cell_id": cell_id, "counters": all_counters}


def generate_timeseries(cell_id: str, n_timesteps: int = 50,
                         degrade_at_step: int = None,
                         degraded_kpis: list[str] = None) -> list[dict]:
    """
    Generate a sequence of counter snapshots for one cell, optionally
    injecting a degradation at a specific timestep (for anomaly-detection
    design discussion / future work — not used in the translation eval).
    """
    series = []
    for t in range(n_timesteps):
        is_degraded = degrade_at_step is not None and t >= degrade_at_step
        snapshot = generate_cell_counters(
            cell_id, degraded_kpis=degraded_kpis if is_degraded else None
        )
        snapshot["timestep"] = t
        series.append(snapshot)
    return series


def verify_registry_consistency() -> dict:
    """
    Sanity check: for each KPI, generate counters targeting the baseline mean,
    compute the KPI back from those counters, and confirm round-trip accuracy.
    Used as a unit test / paper-reportable consistency check.
    """
    results = {}
    for kpi_name, (mean, _) in _KPI_TARGET_DIST.items():
        gen_fn = _KPI_TO_GENERATOR[kpi_name]
        counters = gen_fn(mean)
        computed = compute_kpi(kpi_name, counters)
        error_pct = abs(computed - mean) / mean * 100 if mean else 0
        results[kpi_name] = {
            "target": round(mean, 3),
            "computed": round(computed, 3),
            "error_pct": round(error_pct, 4),
        }
    return results


def generate_pretraining_corpus(n_cells: int = 100, n_timesteps: int = 2000,
                                  anomaly_rate: float = 0.03,
                                  seed: int = 11) -> dict:
    """
    Generate a multivariate KPI time-series corpus suitable for self-supervised
    pretraining: for each of n_cells cells, n_timesteps consecutive readings of
    all 10 registered KPIs (computed from realistic, slowly-drifting PM counters,
    not independent per-timestep noise — see _drift_walk below).

    A small fraction of (cell, timestep) windows are marked as anomalous
    (injected degradation), with ground truth labels retained for downstream
    evaluation (anomaly detection, forecasting) but NOT used during
    self-supervised pretraining itself.

    Returns a dict with:
      "kpi_names": list[str]                     — fixed channel order
      "series": np.ndarray [n_cells, n_timesteps, n_kpis]
      "anomaly_mask": np.ndarray [n_cells, n_timesteps]  (bool)
      "anomaly_kpis": dict[(cell_idx, timestep) -> list[str]]  (which channels degraded)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    kpi_names = list(_KPI_TARGET_DIST.keys())
    n_kpis = len(kpi_names)

    series = np.zeros((n_cells, n_timesteps, n_kpis), dtype=np.float32)
    anomaly_mask = np.zeros((n_cells, n_timesteps), dtype=bool)
    anomaly_kpis: dict = {}

    # Guarantee a minimum number of anomaly bursts proportional to corpus size,
    # rather than relying on a per-segment coin flip that can easily roll zero
    # anomalies at small n_timesteps (e.g. a single cell with only a handful of
    # ~20-60 step segments may never trigger a 3% per-segment draw).
    target_anomaly_fraction = anomaly_rate
    min_bursts_per_cell = max(1, round(n_timesteps * target_anomaly_fraction / 12))  # ~12-step avg burst

    for cell_idx in range(n_cells):
        # Each cell gets its own slowly-drifting baseline per KPI (random walk
        # around the KPI's target mean), so the series has realistic temporal
        # structure (autocorrelation, diurnal-like drift) rather than iid noise.
        current = {k: _KPI_TARGET_DIST[k][0] for k in kpi_names}
        drift_std = {k: _KPI_TARGET_DIST[k][1] * 0.05 for k in kpi_names}  # slow drift

        # Pre-select burst start positions and lengths for this cell up front,
        # so the guaranteed minimum number of anomalies is met exactly,
        # regardless of corpus length (avoids relying on a stepping loop
        # variable to coincide with randomly chosen burst-start indices).
        n_bursts = min_bursts_per_cell
        burst_windows = []  # list of (start, end_exclusive, degraded_kpis)
        if n_timesteps > 40 and n_bursts > 0:
            possible_starts = np.arange(10, n_timesteps - 25)
            chosen_starts = sorted(rng.choice(
                possible_starts, size=min(n_bursts, len(possible_starts)), replace=False
            ).tolist())
            for start in chosen_starts:
                burst_len = int(rng.integers(5, 21))
                degraded = list(rng.choice(kpi_names, size=int(rng.integers(1, 4)), replace=False))
                burst_windows.append((start, min(start + burst_len, n_timesteps), degraded))

        # Build a per-timestep lookup: which (if any) burst is active at t
        active_burst_at = {}
        for start, end, degraded in burst_windows:
            for t_ in range(start, end):
                active_burst_at[t_] = degraded

        for t in range(n_timesteps):
            degraded = active_burst_at.get(t, [])
            for k in kpi_names:
                mean, std = _KPI_TARGET_DIST[k]
                # Random walk step
                current[k] += rng.normal(0, drift_std[k])
                # Mean-revert slightly so values don't wander unboundedly
                current[k] += 0.02 * (mean - current[k])

                value = current[k] + rng.normal(0, std * 0.15)  # observation noise
                if k in degraded:
                    if k in ("PRB_Utilization", "Packet_Loss", "Latency"):
                        value = mean + std * rng.uniform(2.5, 4.0)
                    else:
                        value = mean - std * rng.uniform(2.5, 4.0)

                kpi_idx = kpi_names.index(k)
                series[cell_idx, t, kpi_idx] = max(0.0, value)

            if degraded:
                anomaly_mask[cell_idx, t] = True
                anomaly_kpis[(cell_idx, t)] = degraded

    return {
        "kpi_names": kpi_names,
        "series": series,
        "anomaly_mask": anomaly_mask,
        "anomaly_kpis": anomaly_kpis,
    }


if __name__ == "__main__":
    print("Registry round-trip consistency check:")
    print(f"{'KPI':<20} {'Target':>10} {'Computed':>10} {'Error %':>10}")
    for kpi, r in verify_registry_consistency().items():
        print(f"{kpi:<20} {r['target']:>10} {r['computed']:>10} {r['error_pct']:>10}")

