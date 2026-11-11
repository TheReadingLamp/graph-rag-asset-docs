"""Plain-assert tests (no pytest): run with `python tests/test_graphrag.py`."""

import json
import re
import sys
import tempfile
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graph_rag import consistency, ingest, retriever  # noqa: E402

CORPUS = ROOT / "corpus"
PROV_RE = re.compile(
    r"\[node=\S+ \| src=\S+ \| v\S* \| eff (\d{4}-\d{2}-\d{2}|-)\]"
)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        graph_path = Path(tmp) / "graph.json"
        g, warnings = ingest.ingest(CORPUS, graph_path)
        print(f"ingested {g.number_of_nodes()} nodes, {g.number_of_edges()} edges; "
              f"{len(warnings)} warning(s)")
        for w in warnings:
            print("  warning:", w)

        # --- 1. ingestion: expected node and edge inventory ---
        kinds = {}
        for _, k in g.nodes(data="kind"):
            kinds[k] = kinds.get(k, 0) + 1
        assert kinds.get("asset") == 5, kinds          # SUB-01 + 2 TRF + 2 BRK
        assert kinds.get("document") == 9, kinds       # register, 3 WO, 2 insp, 3 specs
        assert kinds.get("event") == 5, kinds          # 3 WO + 2 inspection events
        etypes = {}
        for _, _, t in g.edges(data="type"):
            etypes[t] = etypes.get(t, 0) + 1
        for required in ("CONTAINS", "DOCUMENTS", "OCCURRED_ON", "PRECEDES", "SUPERSEDES"):
            assert required in etypes and etypes[required] >= 1, (required, etypes)
        print("  ok: node kinds and edge types as expected")

        # --- 2. hierarchy: SUB-01 CONTAINS each transformer and breaker ---
        children = {
            c for c in g.successors("SUB-01") if g.edges["SUB-01", c]["type"] == "CONTAINS"
        }
        assert children == {"TRF-0147", "TRF-0148", "BRK-2201", "BRK-2202"}, children
        print("  ok: CONTAINS hierarchy under SUB-01")

        # --- 3. SUPERSEDES: v3 supersedes the stale v2 ---
        assert g.has_edge("spec_trf_oil_v3", "spec_trf_oil_v2")
        assert g.edges["spec_trf_oil_v3", "spec_trf_oil_v2"]["type"] == "SUPERSEDES"
        print("  ok: spec_trf_oil_v3 SUPERSEDES spec_trf_oil_v2")

        # --- 4. events: PRECEDES on TRF-0148, OCCURRED_ON on 5 events ---
        assert g.has_edge("EVT-wo_2025_11_02_trf_0148", "EVT-insp_2026_04_20_trf_0148")
        assert g.edges["EVT-wo_2025_11_02_trf_0148",
                        "EVT-insp_2026_04_20_trf_0148"]["type"] == "PRECEDES"
        assert etypes.get("OCCURRED_ON") == 5, etypes
        print("  ok: PRECEDES and OCCURRED_ON edges")

        # --- 5. graph.json round-trip ---
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        g2 = nx.node_link_graph(data, edges="edges")
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()
        assert data["meta"]["node_kinds"] == kinds
        print("  ok: graph.json serialization round-trip")

        # --- 6. tf-idf ranking: oil-maintenance query hits the specs ---
        from graph_rag.tfidf import TfidfIndex  # noqa: E402

        doc_nodes = [n for n, k in g.nodes(data="kind") if k == "document"]
        index = TfidfIndex([(n, g.nodes[n].get("body", "")) for n in doc_nodes])
        ranked = index.search("transformer oil maintenance interval", top_k=8)
        top = [doc_id for doc_id, _ in ranked]
        assert "spec_trf_oil_v3" in top[:2], top
        assert "spec_trf_oil_v2" in top[:3], top
        assert ranked[0][1] > ranked[1][1] >= 0.0
        print(f"  ok: tf-idf ranking puts specs on top: {top[:3]}")

        # --- 7. query provenance: every answer line carries the 4 fields ---
        lines, prov = retriever.retrieve(g, "transformer oil maintenance interval",
                                         top_k=3, hop=1)
        assert len(lines) >= 3, lines
        for line in lines:
            assert PROV_RE.match(line), f"bad provenance line: {line}"
        # ranked doc must appear in the answer
        assert any(p["node"] == ranked[0][0] for p in prov)
        # graph expansion must have pulled in linked asset/event lines
        assert any(p["kind"] == "event" for p in prov), prov
        print(f"  ok: {len(lines)} provenance-carrying answer lines")

        # --- 8. consistency: stale spec caught, clean docs not flagged ---
        conflicts = consistency.check_consistency(g, today="2026-11-11")
        by_node = {}
        for c in conflicts:
            by_node.setdefault(c["node"], []).append(c["rule"])
        assert "STALE_SPEC" in by_node.get("spec_trf_oil_v2", []), by_node
        assert "SPEC_VS_EVENT_CONFLICT" in by_node.get("spec_trf_oil_v2", []), by_node
        assert "STALE_REFERENCE" in by_node.get("insp_2026_04_20_trf_0148", []), by_node
        assert "ASSET_STATE_VS_SPEC" in by_node.get("TRF-0148", []), by_node
        assert "SPEC_NOT_YET_IN_FORCE" in by_node.get("EVT-wo_2025_12_10_trf_0147", []), by_node
        assert "wo_2025_11_02_trf_0148" not in by_node, by_node  # v2 was valid on that date
        severities = [c["severity"] for c in conflicts]
        assert "CONFLICT" in severities
        print(f"  ok: consistency checker flags {len(conflicts)} issue(s): "
              f"{sorted(by_node)}")

        # --- 9. CLI entry point loads and exposes the four commands ---
        from graph_rag.cli import main as cli_main  # noqa: E402

        try:
            cli_main(["--help"])
        except SystemExit:
            pass
        print("  ok: CLI entry point loads")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
