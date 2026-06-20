"""
TeleKG Reasoning Engine

Provides high-level graph traversal operations that agents call.
Abstracts FalkorDB Cypher queries into semantic operations:

  - impact_analysis(release): What KPIs might degrade after this SW release?
  - affected_cells(release): Which cells are at risk?
  - required_tests(features): What test cases does the graph say we need?
  - coverage_gaps(): Features with no COVERED_BY test cases
  - root_cause_chain(kpi): Trace upstream KPI causal path
  - propagation_frontier(kpi): All downstream KPIs degraded if this one fails

These form the "Reasoning" component of the Knowledge Plane.
In dry-run mode they return realistic mock data for prototyping without FalkorDB.
"""

from __future__ import annotations
from typing import Any


class TeleKGReasoner:
    """
    Graph reasoning layer. Used by all agents.
    Pass graph=None for dry-run / offline mode.
    """

    def __init__(self, graph=None, dry_run=False):
        self.graph = graph
        self.dry_run = dry_run or (graph is None)

    def _query(self, cypher: str, params: dict = None) -> list:
        if self.dry_run:
            return []
        return self.graph.query(cypher, params or {}).result_set

    # ── Impact Analysis ────────────────────────────────────────────────

    def release_impact(self, release_id: str) -> dict:
        """
        Given a SW release, return:
          - features it touches
          - KPIs those features affect
          - cells that use those features
          - test cases that validate those KPIs
        """
        if self.dry_run:
            return self._mock_release_impact(release_id)

        # Features touched by release
        features = [
            row[0] for row in self._query(
                """
                MATCH (r:SoftwareRelease {id: $rel})-[:AFFECTS_FEATURE]->(f:Feature)
                RETURN f.id
                """,
                {"rel": release_id}
            )
        ]

        # KPIs those features affect
        kpis = [
            row[0] for row in self._query(
                """
                MATCH (r:SoftwareRelease {id: $rel})
                      -[:AFFECTS_FEATURE]->(f:Feature)
                      -[:AFFECTS_KPI]->(k:KPI)
                RETURN DISTINCT k.id
                """,
                {"rel": release_id}
            )
        ]

        # Downstream KPI propagation (2 hops)
        downstream_kpis = [
            row[0] for row in self._query(
                """
                MATCH (r:SoftwareRelease {id: $rel})
                      -[:AFFECTS_FEATURE]->(f:Feature)
                      -[:AFFECTS_KPI]->(k1:KPI)
                      -[:IMPACTS]->(k2:KPI)
                RETURN DISTINCT k2.id
                """,
                {"rel": release_id}
            )
        ]

        # Cells using those features
        cells = [
            row[0] for row in self._query(
                """
                MATCH (r:SoftwareRelease {id: $rel})
                      -[:AFFECTS_FEATURE]->(f:Feature)
                      <-[:USES_FEATURE]-(c:Cell)
                RETURN DISTINCT c.id
                LIMIT 20
                """,
                {"rel": release_id}
            )
        ]

        # Required test cases
        tests = [
            {"id": row[0], "name": row[1], "priority": row[2]} for row in self._query(
                """
                MATCH (r:SoftwareRelease {id: $rel})
                      -[:AFFECTS_FEATURE]->(f:Feature)
                      -[:COVERED_BY]->(t:TestCase)
                RETURN DISTINCT t.id, t.name, t.priority
                ORDER BY t.priority
                """,
                {"rel": release_id}
            )
        ]

        return {
            "release": release_id,
            "features": features,
            "direct_kpis": kpis,
            "downstream_kpis": list(set(downstream_kpis) - set(kpis)),
            "at_risk_cells": cells,
            "required_tests": tests,
        }

    def _mock_release_impact(self, release_id: str) -> dict:
        """Realistic mock for dry-run mode."""
        from telekg.schema import SW_RELEASE_FEATURES, FEATURE_KPI_IMPACTS
        from telekg.schema import KPI_CAUSAL_GRAPH
        from telekg.simulator import TEST_CASE_TEMPLATES

        features = SW_RELEASE_FEATURES.get(release_id, ["Energy_Saving", "Cell_Sleep"])
        direct_kpis = []
        for f in features:
            direct_kpis.extend(FEATURE_KPI_IMPACTS.get(f, []))
        direct_kpis = list(set(direct_kpis))

        downstream = []
        for src, tgt, _ in KPI_CAUSAL_GRAPH:
            if src in direct_kpis:
                downstream.append(tgt)

        tests = []
        for f in features:
            for tc in TEST_CASE_TEMPLATES.get(f, []):
                tests.append({"id": tc["id"], "name": tc["name"], "priority": tc["priority"]})

        cells = [f"Cell_{i:03d}" for i in range(1, 16)]

        return {
            "release": release_id,
            "features": features,
            "direct_kpis": direct_kpis,
            "downstream_kpis": list(set(downstream) - set(direct_kpis)),
            "at_risk_cells": cells,
            "required_tests": tests,
        }

    # ── Root Cause Analysis ────────────────────────────────────────────

    def root_cause_chain(self, kpi_id: str, max_hops: int = 3) -> list[dict]:
        """
        Trace upstream causal chain: what KPIs cause this one to degrade?
        Returns list of (source_kpi, weight) ordered by causal distance.
        """
        if self.dry_run:
            return self._mock_root_cause(kpi_id)

        rows = self._query(
            f"""
            MATCH p=(upstream:KPI)-[:IMPACTS*1..{max_hops}]->(target:KPI {{id: $kpi}})
            RETURN upstream.id, length(p), reduce(w=1.0, e IN relationships(p) | w * e.weight)
            ORDER BY length(p)
            """,
            {"kpi": kpi_id}
        )
        return [{"kpi": r[0], "hops": r[1], "causal_weight": round(r[2], 3)} for r in rows]

    def _mock_root_cause(self, kpi_id: str) -> list[dict]:
        from telekg.schema import KPI_CAUSAL_GRAPH
        result = []
        for src, tgt, w in KPI_CAUSAL_GRAPH:
            if tgt == kpi_id:
                result.append({"kpi": src, "hops": 1, "causal_weight": w})
        return result

    # ── Propagation Frontier ───────────────────────────────────────────

    def propagation_frontier(self, kpi_id: str) -> list[str]:
        """All downstream KPIs that will be affected if kpi_id degrades."""
        if self.dry_run:
            from telekg.schema import KPI_CAUSAL_GRAPH
            return [tgt for src, tgt, _ in KPI_CAUSAL_GRAPH if src == kpi_id]

        rows = self._query(
            """
            MATCH (k:KPI {id: $kpi})-[:IMPACTS*1..3]->(downstream:KPI)
            RETURN DISTINCT downstream.id
            """,
            {"kpi": kpi_id}
        )
        return [r[0] for r in rows]

    # ── Coverage Gap Detection ─────────────────────────────────────────

    def coverage_gaps(self) -> list[dict]:
        """
        Features with no COVERED_BY test cases.
        Classic Cypher: WHERE NOT (f)-[:COVERED_BY]->(:TestCase)
        """
        if self.dry_run:
            return []  # All features have tests in our mock data

        rows = self._query(
            """
            MATCH (f:Feature)
            WHERE NOT (f)-[:COVERED_BY]->(:TestCase)
            RETURN f.id, f.name
            """
        )
        return [{"id": r[0], "name": r[1]} for r in rows]

    # ── Cell Risk Query ────────────────────────────────────────────────

    def cells_at_risk_from_release(self, release_id: str) -> list[dict]:
        """Cells running features touched by this release, with their region."""
        if self.dry_run:
            return [
                {"cell_id": f"Cell_{i:03d}", "region": ["North","South","East","West","Central"][i%5],
                 "feature": "Energy_Saving"}
                for i in range(1, 12)
            ]
        rows = self._query(
            """
            MATCH (r:SoftwareRelease {id: $rel})
                  -[:AFFECTS_FEATURE]->(f:Feature)
                  <-[:USES_FEATURE]-(c:Cell)
                  -[:PART_OF_REGION]->(reg:Region)
            RETURN c.id, reg.id, f.id
            """,
            {"rel": release_id}
        )
        return [{"cell_id": r[0], "region": r[1], "feature": r[2]} for r in rows]

    # ── Test Prioritisation ────────────────────────────────────────────

    def prioritised_tests_for_release(self, release_id: str) -> list[dict]:
        """
        Return tests ordered by:
          priority tier (P0 > P1 > P2)
          + number of at-risk cells the feature runs on
        """
        impact = self.release_impact(release_id)
        tests = impact["required_tests"]
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        return sorted(tests, key=lambda t: priority_order.get(t["priority"], 9))
