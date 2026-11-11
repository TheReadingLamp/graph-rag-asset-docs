"""Ingestion: regex/heuristic entity + relation extraction into a NetworkX property graph.

Node kinds: asset, document, event.
Edge types: CONTAINS, DOCUMENTS, OCCURRED_ON, PRECEDES, SUPERSEDES, REFERENCES.
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import networkx as nx

ASSET_ID_RE = re.compile(r"\b(?:SUB|TRF|BRK)-\d+\b")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
SPEC_NUMBER_RE = re.compile(r"minimum oil dielectric strength:?\s*(\d+(?:\.\d+)?)\s*kv", re.I)


def _load_doc(path):
    """Load a corpus file into a dict with doc_id, doc_type, title, version,
    effective_date, body and any extra fields. Handles .json and .md (front matter)."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        body = data.pop("body", "")
        data["source"] = str(path)
        data["body"] = body
        return data
    text = path.read_text(encoding="utf-8")
    meta = {}
    m = FRONT_MATTER_RE.match(text)
    body = text
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        body = text[m.end():]
    meta["source"] = str(path)
    meta["body"] = body.strip()
    return meta


def _asset_ids(data):
    """Asset ids named in the assets field plus any id-like tokens in the body."""
    ids = set()
    for token in str(data.get("assets", "")).replace(",", " ").split():
        if ASSET_ID_RE.fullmatch(token):
            ids.add(token)
    ids.update(ASSET_ID_RE.findall(data.get("body", "")))
    if data.get("asset_id"):
        ids.add(data["asset_id"])
    return sorted(ids)


def build_graph(corpus_dir):
    """Parse corpus_dir into a NetworkX DiGraph; returns (graph, warnings)."""
    corpus = Path(corpus_dir)
    docs = []
    for path in sorted(corpus.iterdir()):
        if path.suffix in (".json", ".md"):
            docs.append(_load_doc(path))
    warnings = []
    g = nx.DiGraph()

    # Pass 1: asset nodes (register hierarchy + stubs for any id mentioned).
    for d in docs:
        if d.get("doc_type") != "asset_register":
            continue
        for a in d.get("assets", []):
            attrs = {"kind": "asset"}
            attrs.update({k: v for k, v in a.items() if k != "id"})
            g.add_node(a["id"], **attrs)
    known_assets = {n for n, k in g.nodes(data="kind") if k == "asset"}
    for d in docs:
        for aid in _asset_ids(d):
            if aid not in known_assets:
                g.add_node(aid, kind="asset", asset_type="unknown", status="unknown")
                known_assets.add(aid)

    # Pass 2: document nodes.
    for d in docs:
        attrs = {
            "kind": "document",
            "doc_type": d.get("doc_type", "unknown"),
            "title": d.get("title", ""),
            "version": d.get("version"),
            "effective_date": d.get("effective_date") or d.get("date"),
            "source": d["source"],
            "body": d.get("body", ""),
            "work_order_no": d.get("work_order_no"),
            "activity": d.get("activity"),
            "governing_specs": d.get("governing_specs"),
        }
        attrs = {k: v for k, v in attrs.items() if v is not None}
        g.add_node(d["doc_id"], **attrs)

    # Pass 3: CONTAINS edges from the register hierarchy.
    for d in docs:
        if d.get("doc_type") != "asset_register":
            continue
        for a in d.get("assets", []):
            if a.get("parent"):
                g.add_edge(a["parent"], a["id"], type="CONTAINS")

    # Pass 4: DOCUMENTS edges (doc -> node it describes) and event nodes.
    for d in docs:
        doc_id = d["doc_id"]
        for aid in _asset_ids(d):
            if g.has_node(aid):
                g.add_edge(doc_id, aid, type="DOCUMENTS")
        if d.get("doc_type") == "work_order" and d.get("asset_id"):
            event_id = f"EVT-{doc_id}"
            g.add_node(
                event_id,
                kind="event",
                date=d.get("date"),
                activity=d.get("activity", "maintenance"),
                asset_id=d["asset_id"],
                work_order_no=d.get("work_order_no"),
                source=d["source"],
            )
            g.add_edge(doc_id, event_id, type="DOCUMENTS")
            g.add_edge(event_id, d["asset_id"], type="OCCURRED_ON")
        elif d.get("doc_type") == "inspection_report":
            event_id = f"EVT-{doc_id}"
            g.add_node(
                event_id,
                kind="event",
                date=d.get("date"),
                activity="inspection",
                asset_id=(_asset_ids(d)[0] if _asset_ids(d) else None),
                source=d["source"],
            )
            g.add_edge(doc_id, event_id, type="DOCUMENTS")
            if event_id in g and g.nodes[event_id].get("asset_id") and g.has_node(
                g.nodes[event_id]["asset_id"]
            ):
                g.add_edge(event_id, g.nodes[event_id]["asset_id"], type="OCCURRED_ON")

    # Pass 5: SUPERSEDES and REFERENCES edges.
    for d in docs:
        doc_id = d["doc_id"]
        if d.get("supersedes"):
            target = d["supersedes"].strip()
            if g.has_node(target):
                g.add_edge(doc_id, target, type="SUPERSEDES")
            else:
                warnings.append(
                    f"note: {doc_id} supersedes {target} but that document is not "
                    f"in the corpus; edge skipped"
                )
        specs = d.get("governing_specs") or []
        if isinstance(specs, str):
            specs = [s.strip() for s in specs.split(",")]
        for spec_id in specs:
            if g.has_node(spec_id):
                g.add_edge(doc_id, spec_id, type="REFERENCES")

    # Pass 6: PRECEDES between events on the same asset, ordered by date.
    events_by_asset = {}
    for n, kind in g.nodes(data="kind"):
        if kind != "event":
            continue
        aid = g.nodes[n].get("asset_id")
        if aid:
            events_by_asset.setdefault(aid, []).append((g.nodes[n].get("date") or "", n))
    for aid, evs in events_by_asset.items():
        evs.sort()
        for (_, earlier), (_, later) in zip(evs, evs[1:]):
            g.add_edge(earlier, later, type="PRECEDES")

    return g, warnings


def serialize_graph(g, corpus_dir):
    """node_link_data + metadata, ready for graph.json."""
    data = nx.node_link_data(g, edges="edges")
    data["meta"] = {
        "generated": date.today().isoformat(),
        "corpus_dir": str(corpus_dir),
        "node_kinds": _count(g.nodes(data="kind")),
        "edge_types": _count(g.edges(data="type")),
    }
    return data


def _count(pairs):
    counts = {}
    for item in pairs:
        value = item[-1]
        counts[value] = counts.get(value, 0) + 1
    return counts


def ingest(corpus_dir, out_path):
    g, warnings = build_graph(corpus_dir)
    data = serialize_graph(g, corpus_dir)
    Path(out_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return g, warnings


if __name__ == "__main__":
    g, warnings = ingest(sys.argv[1] if len(sys.argv) > 1 else "corpus",
                         sys.argv[2] if len(sys.argv) > 2 else "graph.json")
    for w in warnings:
        print(w, file=sys.stderr)
    print(f"ingested {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
