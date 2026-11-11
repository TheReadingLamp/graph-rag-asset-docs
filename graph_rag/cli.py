"""Argparse CLI: ingest, query, check-consistency, stats."""

import argparse
import sys
from pathlib import Path

from . import consistency, ingest, retriever


def _cmd_ingest(args):
    g, warnings = ingest.ingest(args.corpus, args.out)
    for w in warnings:
        print(w, file=sys.stderr)
    print(f"ingested {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {args.out}")


def _cmd_query(args):
    g = retriever.load_graph(args.graph)
    doc_nodes = [n for n, k in g.nodes(data="kind") if k == "document"]
    from .tfidf import TfidfIndex

    index = TfidfIndex([(n, g.nodes[n].get("body", "")) for n in doc_nodes])
    ranked = index.search(args.query, top_k=args.top_k)
    print(f"Query: {args.query}")
    print("tf-idf ranked documents (cosine):")
    for rank, (doc_id, score) in enumerate(ranked, 1):
        print(f"  {rank}. {doc_id:<32} {score:.4f}")
    lines, _ = retriever.retrieve(g, args.query, top_k=args.top_k, hop=args.hop)
    print(f"Graph-grounded answer (1-hop expansion, every line carries provenance):")
    for line in lines:
        print(f"  {line}")


def _cmd_check(args):
    g = retriever.load_graph(args.graph)
    conflicts = consistency.check_consistency(g, today=args.today)
    for line in consistency.report(conflicts):
        print(line)
    if args.strict and any(c["severity"] == "CONFLICT" for c in conflicts):
        sys.exit(1)


def _cmd_stats(args):
    g = retriever.load_graph(args.graph)
    kinds = {}
    for _, k in g.nodes(data="kind"):
        kinds[k] = kinds.get(k, 0) + 1
    etypes = {}
    for _, _, t in g.edges(data="type"):
        etypes[t] = etypes.get(t, 0) + 1
    dtypes = {}
    for n, k in g.nodes(data="kind"):
        if k == "document":
            dt = g.nodes[n].get("doc_type", "?")
            dtypes[dt] = dtypes.get(dt, 0) + 1
    print(f"nodes: {g.number_of_nodes()}  edges: {g.number_of_edges()}")
    print("node kinds:", ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print("edge types:", ", ".join(f"{k}={v}" for k, v in sorted(etypes.items())))
    print("document types:", ", ".join(f"{k}={v}" for k, v in sorted(dtypes.items())))
    print("asset hierarchy (CONTAINS):")
    contained = {v for _, v, t in g.edges(data="type") if t == "CONTAINS"}
    roots = [
        n for n in g.nodes if g.nodes[n].get("kind") == "asset" and n not in contained
    ]
    for root in sorted(roots):
        print(f"  {root}")
        for child in sorted(g.successors(root)):
            if g.edges[root, child].get("type") == "CONTAINS":
                print(f"    |- {child} ({g.nodes[child].get('asset_type', '?')}, "
                      f"{g.nodes[child].get('status', '?')})")
    print("events per asset:")
    for n, k in sorted(g.nodes(data="kind")):
        if k == "asset":
            evs = [p for p in g.predecessors(n)
                   if g.edges[p, n].get("type") == "OCCURRED_ON"]
            print(f"  {n}: {len(evs)} event(s)" + (" -> " + ", ".join(sorted(evs)) if evs else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="graph-rag-asset-docs",
        description="Graph-grounded RAG over utility asset-document networks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="extract nodes/edges from corpus/ into graph.json")
    p.add_argument("--corpus", default="corpus", help="corpus directory (default: corpus)")
    p.add_argument("--out", default="graph.json", help="output graph file (default: graph.json)")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("query", help="rank documents by tf-idf, expand 1 hop, answer with provenance")
    p.add_argument("query", help="natural-language query")
    p.add_argument("--graph", default="graph.json")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--hop", type=int, default=1)
    p.set_defaults(func=_cmd_query)

    p = sub.add_parser("check-consistency", help="flag stale specs and temporal conflicts")
    p.add_argument("--graph", default="graph.json")
    p.add_argument("--today", default=None, help="as-of date (YYYY-MM-DD), default: system date")
    p.add_argument("--strict", action="store_true", help="exit 1 if any CONFLICT is found")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("stats", help="print graph statistics and asset hierarchy")
    p.add_argument("--graph", default="graph.json")
    p.set_defaults(func=_cmd_stats)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
