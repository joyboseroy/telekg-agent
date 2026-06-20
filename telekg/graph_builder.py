"""
TeleKG Graph Builder

Ingests the synthetic network dataset into FalkorDB, creating the full
three-layer Knowledge Graph:

  Layer 1 — Network topology (cells, DUs, clusters, regions)
  Layer 2 — KPI causal graph (KPI nodes + IMPACTS edges)
  Layer 3 — Feature/software/test graph

Then connects the layers:
  Cell → HAS_KPI → KPI
  Cell → USES_FEATURE → Feature
  SoftwareRelease → AFFECTS_FEATURE → Feature
  Feature → AFFECTS_KPI → KPI
  Feature → COVERED_BY → TestCase
  TestCase → VALIDATES → KPI

Usage:
    python -m telekg.graph_builder [--reset]

Requires FalkorDB running:
    docker run -p 6379:6379 falkordb/falkordb:latest
"""

import sys
import json
import pathlib
from typing import Any

try:
    import falkordb
    FALKOR_AVAILABLE = True
except ImportError:
    FALKOR_AVAILABLE = False
    print("[WARN] falkordb not installed — running in dry-run/print mode")

from telekg.schema import (
    FEATURE_KPI_IMPACTS, SW_RELEASE_FEATURES, KPI_CAUSAL_GRAPH,
    ALL_KPIS,
)
from telekg.simulator import TEST_CASE_TEMPLATES


GRAPH_NAME = "telekg"


class TeleKGBuilder:
    """Builds and queries the Telecom Knowledge Graph in FalkorDB."""

    def __init__(self, host="localhost", port=6379, dry_run=False):
        self.dry_run = dry_run or not FALKOR_AVAILABLE
        self._queries_run = []
        if not self.dry_run:
            self.db = falkordb.FalkorDB(host=host, port=port)
            self.graph = self.db.select_graph(GRAPH_NAME)
        else:
            self.graph = None

    def _run(self, query: str, params: dict = None):
        """Execute a Cypher query, or log it in dry-run mode."""
        params = params or {}
        if self.dry_run:
            self._queries_run.append((query, params))
            return []
        result = self.graph.query(query, params)
        return result.result_set

    def reset(self):
        """Drop and recreate the graph."""
        if not self.dry_run:
            try:
                self.db.select_graph(GRAPH_NAME).delete()
            except Exception:
                pass
            self.graph = self.db.select_graph(GRAPH_NAME)
        print("[INFO] Graph reset.")

    # ── Layer 1: Network Topology ──────────────────────────────────────

    def create_topology(self, cells: list[dict]):
        regions_seen = set()
        clusters_seen = set()
        dus_seen = set()

        for cell in cells:
            region = cell["region"]
            du_id = cell["du_id"]
            cluster = cell["cluster"]
            cell_id = cell["cell_id"]

            # Region node
            if region not in regions_seen:
                self._run(
                    "MERGE (:Region {id: $id, name: $id})",
                    {"id": region}
                )
                regions_seen.add(region)

            # K8s Cluster node
            if cluster not in clusters_seen:
                self._run(
                    "MERGE (:K8sCluster {id: $id, name: $id})",
                    {"id": cluster}
                )
                clusters_seen.add(cluster)

            # DU node
            if du_id not in dus_seen:
                self._run(
                    "MERGE (:DistributedUnit {id: $id})",
                    {"id": du_id}
                )
                # DU hosted on cluster
                self._run(
                    """
                    MATCH (du:DistributedUnit {id: $du})
                    MATCH (k:K8sCluster {id: $cluster})
                    MERGE (du)-[:HOSTED_ON]->(k)
                    """,
                    {"du": du_id, "cluster": cluster}
                )
                dus_seen.add(du_id)

            # Cell node
            self._run(
                """
                MERGE (c:Cell {id: $id})
                SET c.region = $region, c.lat = $lat, c.lon = $lon
                """,
                {"id": cell_id, "region": region,
                 "lat": cell.get("lat", 0), "lon": cell.get("lon", 0)}
            )
            # Cell → Region
            self._run(
                """
                MATCH (c:Cell {id: $cell})
                MATCH (r:Region {id: $region})
                MERGE (c)-[:PART_OF_REGION]->(r)
                """,
                {"cell": cell_id, "region": region}
            )
            # Cell → DU
            self._run(
                """
                MATCH (c:Cell {id: $cell})
                MATCH (du:DistributedUnit {id: $du})
                MERGE (c)-[:SERVED_BY]->(du)
                """,
                {"cell": cell_id, "du": du_id}
            )

        print(f"[INFO] Topology: {len(cells)} cells, {len(dus_seen)} DUs, "
              f"{len(clusters_seen)} clusters, {len(regions_seen)} regions")

    # ── Layer 2: KPI Causal Graph ──────────────────────────────────────

    def create_kpi_graph(self):
        kpis_created = set()
        for kpi_from, kpi_to, weight in KPI_CAUSAL_GRAPH:
            for kpi in [kpi_from, kpi_to]:
                if kpi not in kpis_created:
                    self._run(
                        "MERGE (:KPI {id: $id, name: $id})",
                        {"id": kpi}
                    )
                    kpis_created.add(kpi)
            self._run(
                """
                MATCH (a:KPI {id: $from})
                MATCH (b:KPI {id: $to})
                MERGE (a)-[:IMPACTS {weight: $w}]->(b)
                """,
                {"from": kpi_from, "to": kpi_to, "w": weight}
            )

        print(f"[INFO] KPI graph: {len(kpis_created)} KPIs, "
              f"{len(KPI_CAUSAL_GRAPH)} causal edges")

    # ── Layer 3: Feature / Software / Test Graph ───────────────────────

    def create_feature_graph(self):
        # Features
        for feature, kpis in FEATURE_KPI_IMPACTS.items():
            self._run(
                "MERGE (:Feature {id: $id, name: $id})",
                {"id": feature}
            )
            for kpi in kpis:
                self._run(
                    """
                    MATCH (f:Feature {id: $feat})
                    MATCH (k:KPI {id: $kpi})
                    MERGE (f)-[:AFFECTS_KPI]->(k)
                    """,
                    {"feat": feature, "kpi": kpi}
                )

        # Software releases
        for release, features in SW_RELEASE_FEATURES.items():
            self._run(
                "MERGE (:SoftwareRelease {id: $id, name: $id})",
                {"id": release}
            )
            for feat in features:
                self._run(
                    """
                    MATCH (r:SoftwareRelease {id: $rel})
                    MATCH (f:Feature {id: $feat})
                    MERGE (r)-[:AFFECTS_FEATURE]->(f)
                    """,
                    {"rel": release, "feat": feat}
                )

        # Test cases
        tc_count = 0
        for feature, tcs in TEST_CASE_TEMPLATES.items():
            for tc in tcs:
                self._run(
                    """
                    MERGE (t:TestCase {id: $id})
                    SET t.name = $name, t.priority = $priority
                    """,
                    {"id": tc["id"], "name": tc["name"], "priority": tc["priority"]}
                )
                # Feature → COVERED_BY → TestCase
                self._run(
                    """
                    MATCH (f:Feature {id: $feat})
                    MATCH (t:TestCase {id: $tc})
                    MERGE (f)-[:COVERED_BY]->(t)
                    """,
                    {"feat": feature, "tc": tc["id"]}
                )
                # TestCase → VALIDATES → KPI
                for kpi in tc["validates"]:
                    self._run(
                        """
                        MATCH (t:TestCase {id: $tc})
                        MATCH (k:KPI {id: $kpi})
                        MERGE (t)-[:VALIDATES]->(k)
                        """,
                        {"tc": tc["id"], "kpi": kpi}
                    )
                tc_count += 1

        print(f"[INFO] Feature graph: {len(FEATURE_KPI_IMPACTS)} features, "
              f"{len(SW_RELEASE_FEATURES)} releases, {tc_count} test cases")

    # ── Cross-layer: Cell ↔ KPI ↔ Feature ─────────────────────────────

    def link_cells_to_features_and_kpis(self, cells: list[dict]):
        """Connect each cell to its active features and instantiate KPI nodes per cell."""
        for cell in cells:
            cell_id = cell["cell_id"]
            for feature in cell.get("active_features", []):
                self._run(
                    """
                    MATCH (c:Cell {id: $cell})
                    MATCH (f:Feature {id: $feat})
                    MERGE (c)-[:USES_FEATURE]->(f)
                    """,
                    {"cell": cell_id, "feat": feature}
                )
            # HAS_KPI edges: one per KPI for this cell
            for kpi in ALL_KPIS:
                self._run(
                    """
                    MATCH (c:Cell {id: $cell})
                    MATCH (k:KPI {id: $kpi})
                    MERGE (c)-[:HAS_KPI]->(k)
                    """,
                    {"cell": cell_id, "kpi": kpi}
                )

        print(f"[INFO] Cross-layer edges added for {len(cells)} cells")

    # ── Change Event Ingestion ─────────────────────────────────────────

    def ingest_change_event(self, event: dict):
        """Record a change event and link it to affected cells and triggers."""
        self._run(
            """
            MERGE (e:ChangeEvent {id: $id})
            SET e.type = $type, e.trigger = $trigger,
                e.timestamp = $ts, e.description = $desc
            """,
            {
                "id": event["event_id"],
                "type": event["event_type"],
                "trigger": event["trigger"],
                "ts": event["timestamp"],
                "desc": event["description"],
            }
        )
        for cell_id in event["affected_cells"]:
            self._run(
                """
                MATCH (e:ChangeEvent {id: $eid})
                MATCH (c:Cell {id: $cell})
                MERGE (e)-[:AFFECTS_CELL]->(c)
                """,
                {"eid": event["event_id"], "cell": cell_id}
            )
        # If SW_UPGRADE, link to the release node
        if event["event_type"] == "SW_UPGRADE":
            self._run(
                """
                MATCH (e:ChangeEvent {id: $eid})
                MERGE (r:SoftwareRelease {id: $rel})
                MERGE (e)-[:CAUSED_BY]->(r)
                """,
                {"eid": event["event_id"], "rel": event["trigger"]}
            )

    # ── Graph Statistics ───────────────────────────────────────────────

    def stats(self) -> dict:
        if self.dry_run:
            return {"mode": "dry_run", "queries_recorded": len(self._queries_run)}
        counts = {}
        for label in ["Cell", "KPI", "Feature", "SoftwareRelease", "TestCase",
                      "DistributedUnit", "K8sCluster", "Region", "ChangeEvent"]:
            r = self._run(f"MATCH (n:{label}) RETURN count(n)")
            counts[label] = r[0][0] if r else 0
        return counts


def build_graph(data_path="data/synthetic_network.json", reset=True, dry_run=False):
    """Full pipeline: load data → build all graph layers."""
    data = json.loads(pathlib.Path(data_path).read_text())
    builder = TeleKGBuilder(dry_run=dry_run)

    if reset:
        builder.reset()

    print("[INFO] Building Layer 1: Network Topology...")
    builder.create_topology(data["cells"])

    print("[INFO] Building Layer 2: KPI Causal Graph...")
    builder.create_kpi_graph()

    print("[INFO] Building Layer 3: Feature/Software/Test Graph...")
    builder.create_feature_graph()

    print("[INFO] Adding cross-layer cell↔feature↔KPI edges...")
    builder.link_cells_to_features_and_kpis(data["cells"])

    print("[INFO] Ingesting change events...")
    for event in data["change_events"]:
        builder.ingest_change_event(event)

    print("[INFO] Graph construction complete.")
    if not dry_run:
        stats = builder.stats()
        print("[INFO] Node counts:", stats)

    return builder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print queries, don't run")
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    # Generate data first if needed
    data_file = pathlib.Path("data/synthetic_network.json")
    if not data_file.exists():
        from telekg.simulator import build_full_dataset
        data = build_full_dataset()
        data_file.parent.mkdir(exist_ok=True)
        data_file.write_text(json.dumps(data, indent=2))
        print(f"[INFO] Generated synthetic data: {data_file}")

    build_graph(reset=not args.no_reset, dry_run=args.dry_run)
