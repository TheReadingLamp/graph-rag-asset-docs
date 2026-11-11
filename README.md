# graph-rag-asset-docs

A graph-grounded RAG (retrieval-augmented generation) demo for **utility
asset-document networks**: an asset register, maintenance work orders,
inspection reports and versioned engineering specifications are ingested into
a NetworkX property graph, queried with a from-scratch numpy tf-idf retriever,
and answered with **full provenance on every line**. A temporal-validity
checker flags stale specifications whose version contradicts the latest
maintenance events and the current asset state.

Fully offline. Only two dependencies: `networkx` and `numpy` (tf-idf is
implemented by hand in numpy - no scikit-learn). No LLM API keys are needed;
the answer assembler is deterministic, and the README explains where an LLM
would plug in.

## Architecture

```
                    corpus/  (9 synthetic utility documents)
                       |
                       v
              +-------------------+
              |  INGEST (regex +  |
              |  heuristics, no   |
              |  ML)              |
              +-------------------+
                       |
        nodes: asset / document / event
        edges: CONTAINS, DOCUMENTS, OCCURRED_ON,
               PRECEDES, SUPERSEDES, REFERENCES
                       |
                       v
                 graph.json  (NetworkX property graph)
                 /        |               \
                v         v                v
        +---------+  +----------+  +------------------+
        |  QUERY  |  |  STATS   |  | CHECK-CONSISTENCY|
        | tf-idf  |  | graph    |  | temporal-validity|
        | (numpy) |  | inventory|  | conflict report  |
        +---------+  +----------+  +------------------+
                |
        1-hop graph expansion
        (assets, events, superseding/superseded specs)
                |
                v
        provenance-tracked answer
        [node | src | version | effective date] on every line
```

The deliberate data flaws: (1) `corpus/spec_trf_oil_v2.md` (v2, eff 2025-01-01:
min oil dielectric 40 kV, 6-year bushing interval) remains in the corpus even
though `spec_trf_oil_v3.md` (v3, eff 2026-01-01: 45 kV, 3-year interval)
superseded it, and the latest maintenance event on TRF-0147 (2026-05-14) was
performed under v3; (2) `corpus/wo_2025_12_10_trf_0147.json` records a test
performed on 2025-12-10 under v3 while it was still only in draft - before its
effective date. The consistency checker must catch both - and does.

## Requirements and setup (Windows, Python 3.14)

```
cd D:\Ai\Claude_Code\aria\graph-rag-asset-docs
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verified with Python 3.14.6, networkx 3.7, numpy 2.5.4.

## Usage

```
.venv\Scripts\python.exe -m graph_rag ingest                  # corpus/ -> graph.json
.venv\Scripts\python.exe -m graph_rag query "transformer oil maintenance interval"
.venv\Scripts\python.exe -m graph_rag check-consistency
.venv\Scripts\python.exe -m graph_rag stats
.venv\Scripts\python.exe tests\test_graphrag.py               # plain asserts, no pytest
```

### Sample output: query

```
$ .venv\Scripts\python.exe -m graph_rag query "transformer oil maintenance interval"
Query: transformer oil maintenance interval
tf-idf ranked documents (cosine):
  1. spec_trf_oil_v2                  0.6827
  2. spec_trf_oil_v3                  0.5358
  3. wo_2026_05_14_trf_0147           0.4843
Graph-grounded answer (1-hop expansion, every line carries provenance):
  [node=spec_trf_oil_v2 | src=corpus\spec_trf_oil_v2.md | v2 | eff 2025-01-01] 4.3 Oil change interval: 12 years, or on condition.
  [node=spec_trf_oil_v3 | src=corpus\spec_trf_oil_v3.md | v3 | eff 2026-01-01] 4.3 Oil change interval: 10 years, or on condition.
  [node=wo_2026_05_14_trf_0147 | src=corpus\wo_2026_05_14_trf_0147.json | v- | eff 2026-05-14] Replaced HV and LV bushings and changed 2400 litres of mineral oil on transformer TRF-0147.
  [node=EVT-wo_2025_12_10_trf_0147 | src=corpus\wo_2025_12_10_trf_0147.json | v- | eff 2025-12-10] test event on asset TRF-0147 (2025-12-10), linked in the graph
  [node=insp_2026_04_20_trf_0148 | src=corpus\insp_2026_04_20_trf_0148.md | v- | eff 2026-04-20] Asset inspected: TRF-0148 (132/33 kV power transformer T2).
  [node=EVT-insp_2026_04_20_trf_0148 | src=corpus\insp_2026_04_20_trf_0148.md | v- | eff 2026-04-20] inspection event on asset TRF-0148 (2026-04-20), linked in the graph
  [node=EVT-wo_2025_11_02_trf_0148 | src=corpus\wo_2025_11_02_trf_0148.json | v- | eff 2025-11-02] test event on asset TRF-0148 (2025-11-02), linked in the graph
  [node=wo_2025_12_10_trf_0147 | src=corpus\wo_2025_12_10_trf_0147.json | v- | eff 2025-12-10] Insulation testing on transformer TRF-0147 carried out per spec_trf_oil_v3 as circulated in draft ahead of its formal effective date.
  [node=asset_register | src=corpus\asset_register.json | v7 | eff 2026-03-01] Transformer TRF-0147 oil dielectric strength 46 kV, bushings last tested 2026-05-14.
  [node=EVT-wo_2026_05_14_trf_0147 | src=corpus\wo_2026_05_14_trf_0147.json | v- | eff 2026-05-14] maintenance event on asset TRF-0147 (2026-05-14), linked in the graph
```

Note that the graph expansion pulled the superseded v2 spec *and* the
maintenance event it conflicts with into the answer - exactly the evidence an
LLM would need to say "v2 is stale".

### Sample output: check-consistency

```
$ .venv\Scripts\python.exe -m graph_rag check-consistency
Temporal-validity check
========================
1. [CONFLICT] SPEC_NOT_YET_IN_FORCE -> EVT-wo_2025_12_10_trf_0147
   EVT-wo_2025_12_10_trf_0147 (2025-12-10) was performed under spec_trf_oil_v3 (v3, eff 2026-01-01), but that specification was not yet effective on the event date, so the work and any results recorded against it are not in force.
2. [CONFLICT] ASSET_STATE_VS_SPEC -> TRF-0148
   TRF-0148 is registered with oil dielectric strength 42 kV (asset register in force), which is below the 45 kV minimum of the governing standard spec_trf_oil_v3 (v3, eff 2026-01-01).
3. [CONFLICT] STALE_REFERENCE -> insp_2026_04_20_trf_0148
   insp_2026_04_20_trf_0148 (2026-04-20) cites spec_trf_oil_v2 as the governing standard, but spec_trf_oil_v3 was already effective on that date (eff 2026-01-01), so the reference and any conclusions based on it are stale.
4. [CONFLICT] SPEC_VS_EVENT_CONFLICT -> spec_trf_oil_v2
   spec_trf_oil_v2 (v2) still specifies min oil dielectric 40 kV vs 45 kV; bushing interval 6y vs 3y, but the latest maintenance event on TRF-0147 (EVT-wo_2026_05_14_trf_0147, 2026-05-14) was performed under spec_trf_oil_v3 (v3), so the two documents contradict each other.
5. [STALE] STALE_SPEC -> spec_trf_oil_v2
   spec_trf_oil_v2 (v2, eff 2025-01-01) has been superseded by spec_trf_oil_v3 (v3, eff 2026-01-01) but is still present in the corpus and must not govern current work.

Summary: CONFLICT=4, STALE=1
```

The stale spec is caught twice - as a superseded document and as a version
contradicting the latest maintenance event - and the draft-v3 work order is
caught by the in-force rule, while `wo_2025_11_02_trf_0148` (which correctly
cited v2 while v2 was still in force) is correctly *not* flagged. Use
`check-consistency --today YYYY-MM-DD` to evaluate against a fixed as-of date;
the checker is fully deterministic given an explicit `--today`.

### Sample output: stats

```
$ .venv\Scripts\python.exe -m graph_rag stats
nodes: 19  edges: 38
node kinds: asset=5, document=9, event=5
edge types: CONTAINS=4, DOCUMENTS=21, OCCURRED_ON=5, PRECEDES=2, REFERENCES=5, SUPERSEDES=1
document types: asset_register=1, engineering_spec=3, inspection_report=2, work_order=3
asset hierarchy (CONTAINS):
  SUB-01
    |- BRK-2201 (breaker, in_service)
    |- BRK-2202 (breaker, in_service)
    |- TRF-0147 (transformer, in_service)
    |- TRF-0148 (transformer, in_service)
```

## Why this matters

Utility operations run on **asset-document networks**: a substation
*contains* transformers and breakers; work orders *document* maintenance
events that *occurred on* those assets; engineering specifications *supersede*
each other with version numbers and effective dates. Plain vector RAG flattens
all of that into embeddings and loses containment, event ordering and version
chains. This project keeps the structure: retrieval ranks documents with
classic tf-idf, then the graph supplies the missing relational context
(which asset a claim is about, which events touched it, which version was in
force when) before any text is assembled into an answer.

That makes it a concrete foundation for **graph-grounded LLM reasoning**:
instead of asking a model to answer from embeddings alone, the retriever hands
it a small, verifiable subgraph with **provenance tracking** (node id, source
file, version, effective date) attached to every evidence line, so the model
can cite its sources and refuse to answer when evidence conflicts. The
**temporal-validity** checker is the first step toward reasoning that is
time-aware: it rejects answers that would silently apply a superseded
standard to current work. Extending the graph with live sensor/SCADA state
per asset turns this into a **digital twin** where documentation, event
history and real-time condition share one queryable model - the natural next
milestone of this line of research.

## Where an LLM would plug in

The current answer assembler is deterministic: ranked snippets plus
provenance lines. To add an LLM, replace the assembly step in
`graph_rag/retriever.py` (`retrieve()`) with a model call whose prompt is the
retrieved subgraph (documents + assets + events + version edges) and require
the model to answer only by citing the supplied `[node=... | src=... | v... |
eff ...]` identifiers. No API key is needed to run anything in this repo; the
graph and the tf-idf index work fully offline, and a local model (e.g. Ollama)
can be substituted without code changes elsewhere.

## Graph model

| Node kind | Examples                                    | Key attributes                    |
|-----------|---------------------------------------------|-----------------------------------|
| asset     | SUB-01, TRF-0147, BRK-2201                  | asset_type, status, name          |
| document  | spec_trf_oil_v3, wo_2026_05_14_trf_0147     | doc_type, version, effective_date, source, body |
| event     | EVT-wo_2026_05_14_trf_0147                  | date, activity, asset_id          |

| Edge type  | Meaning                                                        |
|------------|----------------------------------------------------------------|
| CONTAINS   | physical containment in the asset hierarchy                    |
| DOCUMENTS  | a document describes a node (asset, event, ...)                |
| OCCURRED_ON| a maintenance/inspection event happened on an asset            |
| PRECEDES   | chronological order of events on the same asset                |
| SUPERSEDES | a newer specification version replaces an older one            |
| REFERENCES | a work order / report cites a governing specification          |

## Corpus

`corpus/` holds 9 synthetic documents:

- `asset_register.json` - asset hierarchy (SUB-01 -> TRF-0147/0148, BRK-2201/2202), v7, eff 2026-03-01
- `wo_2026_05_14_trf_0147.json` - oil change + bushing replacement, performed under spec v3
- `wo_2025_12_10_trf_0147.json` - bushing test performed under v3 **before it was in force** (draft use; deliberately planted)
- `wo_2025_11_02_trf_0148.json` - bushing test, correctly performed under v2 (still in force then)
- `insp_2026_04_20_trf_0148.md` - inspection that *stale-cites* v2 after v3 became effective
- `insp_2026_09_30_brk_2201.md` - clean post-refill inspection under spec_breaker_sf6_v1
- `spec_trf_oil_v2.md` - the **deliberately stale specification** (superseded, contradicts latest event)
- `spec_trf_oil_v3.md` - current version, supersedes v2
- `spec_breaker_sf6_v1.md` - breaker gas-handling spec, v1

## Project layout

```
graph-rag-asset-docs/
|-- corpus/                  # 9 synthetic utility documents (.md / .json)
|-- graph_rag/
|   |-- __init__.py
|   |-- __main__.py
|   |-- ingest.py            # regex/heuristic extraction -> NetworkX graph -> graph.json
|   |-- tfidf.py             # from-scratch numpy tf-idf (no scikit-learn)
|   |-- retriever.py         # tf-idf ranking + 1-hop expansion + provenance answer
|   |-- consistency.py       # temporal-validity conflict checker
|   `-- cli.py               # argparse: ingest, query, check-consistency, stats
|-- tests/
|   `-- test_graphrag.py     # plain-assert tests (no pytest): python tests\test_graphrag.py
|-- graph.json               # generated by ingest (checked in for convenience)
|-- requirements.txt
`-- README.md
```

## Tests

`python tests\test_graphrag.py` runs 9 assertion groups covering ingestion
inventory, the CONTAINS hierarchy, SUPERSEDES edges, PRECEDES/OCCURRED_ON,
graph.json round-tripping, tf-idf ranking, provenance formatting on every
answer line, conflict detection (including the negative case that must not
fire), and the CLI entry point. No pytest required.
