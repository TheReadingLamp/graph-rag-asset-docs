"""Graph-grounded retrieval: numpy tf-idf ranking + 1-hop graph expansion +
provenance-tracked answer assembly. Fully offline; no LLM API required."""

import json
import re
from pathlib import Path

import networkx as nx

from .tfidf import TfidfIndex, tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def load_graph(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges")


def _provenance(g, node_id):
    """The four provenance fields every answer line must carry."""
    n = g.nodes[node_id]
    version = n.get("version") or "-"
    eff = n.get("effective_date") or n.get("date") or "-"
    src = n.get("source", "-")
    return f"[node={node_id} | src={src} | v{version} | eff {eff}]"


def _snippet(text, query_terms, max_len=220):
    """Best sentence(s) containing query terms, else the opening sentence."""
    cleaned = "\n".join(
        line for line in (text or "").splitlines() if not line.lstrip().startswith("#")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = [s.strip() for s in _SENT_RE.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned[:max_len]
    qset = set(query_terms)
    scored = sorted(sentences, key=lambda s: -len(qset & set(tokenize(s))))
    best = scored[0]
    if not (qset & set(tokenize(best))) and len(sentences) > 1:
        best = sentences[0]
    return best[:max_len]


def retrieve(g, query, top_k=3, hop=1, max_context=6):
    """Rank document nodes by tf-idf, expand over the graph, assemble an answer.

    Returns (answer_lines, provenance_list). Every answer line starts with a
    provenance bracket carrying node id, source file, version and effective date.
    """
    doc_nodes = [n for n, k in g.nodes(data="kind") if k == "document"]
    index = TfidfIndex([(n, g.nodes[n].get("body", "")) for n in doc_nodes])
    ranked = index.search(query, top_k=top_k)
    query_terms = [t for t in tokenize(query)]

    # 1-hop expansion from the ranked documents over the graph.
    expanded = set()
    frontier = [doc_id for doc_id, _ in ranked]
    if hop >= 1:
        for doc_id in list(frontier):
            for nbr in g.neighbors(doc_id):
                if g.nodes[nbr].get("kind") != "document":
                    frontier.append(nbr)
            for pred in g.predecessors(doc_id):
                if g.nodes[pred].get("kind") != "document":
                    frontier.append(pred)
    for node_id in frontier:
        if g.nodes[node_id].get("kind") == "asset":
            # ground the answer in the asset network: sibling docs + events
            for doc in g.predecessors(node_id):
                if g.edges[doc, node_id].get("type") == "DOCUMENTS":
                    expanded.add(doc)
            for ev in g.predecessors(node_id):
                if g.edges[ev, node_id].get("type") == "OCCURRED_ON":
                    expanded.add(ev)
        elif g.nodes[node_id].get("kind") == "event":
            expanded.add(node_id)
            for doc in g.predecessors(node_id):
                if g.edges[doc, node_id].get("type") == "DOCUMENTS":
                    expanded.add(doc)

    ordered = [d for d, _ in ranked]
    for node_id in expanded:
        if node_id not in ordered:
            ordered.append(node_id)

    lines = []
    prov = []
    shown = 0
    for node_id in ordered:
        kind = g.nodes[node_id].get("kind")
        if kind == "document":
            if shown >= max_context and node_id not in {d for d, _ in ranked}:
                continue
            p = _provenance(g, node_id)
            lines.append(f"{p} {_snippet(g.nodes[node_id].get('body', ''), query_terms)}")
            prov.append({"node": node_id, "kind": "document", "provenance": p})
            shown += 1
        elif kind == "event":
            n = g.nodes[node_id]
            p = _provenance(g, node_id)
            aid = n.get("asset_id", "?")
            lines.append(
                f"{p} {n.get('activity', 'event')} event on asset {aid} "
                f"({n.get('date', '-')}), linked in the graph"
            )
            prov.append({"node": node_id, "kind": "event", "provenance": p})
    return lines, prov


if __name__ == "__main__":
    import sys

    g = load_graph(sys.argv[2] if len(sys.argv) > 2 else "graph.json")
    lines, _ = retrieve(g, sys.argv[1])
    for line in lines:
        print(line)
