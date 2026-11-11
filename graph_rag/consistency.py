"""Temporal-validity checker: flags documents whose effective version conflicts
with the latest maintenance events or the current asset state."""

import re
from datetime import date

import networkx as nx

from .retriever import load_graph

MIN_DIELECTRIC_RE = re.compile(
    r"minimum oil dielectric strength:?\s*(\d+(?:\.\d+)?)\s*kv", re.I
)
BUSING_INTERVAL_RE = re.compile(
    r"bushing insulation test interval:?\s*(\d+(?:\.\d+)?)\s*years?", re.I
)


def _iso(d):
    return str(d or "")


def _assets_of(g, doc_id):
    return sorted(
        n for n in g.neighbors(doc_id) if g.edges[doc_id, n].get("type") == "DOCUMENTS"
    )


def _referenced_specs(g, doc_id):
    return sorted(
        n for n in g.neighbors(doc_id) if g.edges[doc_id, n].get("type") == "REFERENCES"
    )


def _superseding_specs(g, doc_id):
    return sorted(
        p for p in g.predecessors(doc_id) if g.edges[p, doc_id].get("type") == "SUPERSEDES"
    )


def _source_doc_of_event(g, event_id):
    for doc in g.predecessors(event_id):
        if g.edges[doc, event_id].get("type") == "DOCUMENTS":
            return doc
    return None


def _doc_date(g, node_id):
    return g.nodes[node_id].get("effective_date") or g.nodes[node_id].get("date") or ""


def check_consistency(g, today=None):
    """Return a list of conflict dicts: {rule, node, severity, message}."""
    today = today or date.today().isoformat()
    conflicts = []

    # Rule 1: superseded specs still present in the corpus and already out of force.
    for doc_id in sorted(g.nodes):
        if g.nodes[doc_id].get("kind") != "document":
            continue
        for newer in _superseding_specs(g, doc_id):
            newer_eff = _doc_date(g, newer)
            if newer_eff and newer_eff <= today:
                old_v, new_v = g.nodes[doc_id].get("version"), g.nodes[newer].get("version")
                conflicts.append({
                    "rule": "STALE_SPEC",
                    "node": doc_id,
                    "severity": "STALE",
                    "message": (
                        f"{doc_id} (v{old_v}, eff {_doc_date(g, doc_id)}) has been "
                        f"superseded by {newer} (v{new_v}, eff {newer_eff}) but is "
                        f"still present in the corpus and must not govern current work."
                    ),
                })

    # Rule 2: a superseded spec contradicts the latest maintenance event on an
    # asset it covers (the event was performed under the newer spec).
    events = sorted(n for n, k in g.nodes(data="kind") if k == "event")
    latest_by_asset = {}
    for ev in events:
        aid = g.nodes[ev].get("asset_id")
        if aid:
            if aid not in latest_by_asset or (g.nodes[ev].get("date") or "") > (
                g.nodes[latest_by_asset[aid]].get("date") or ""
            ):
                latest_by_asset[aid] = ev
    for aid, ev_id in latest_by_asset.items():
        if g.nodes[ev_id].get("activity") != "maintenance":
            continue
        ev_date = g.nodes[ev_id].get("date") or ""
        src_doc = _source_doc_of_event(g, ev_id)
        if not src_doc:
            continue
        used_specs = _referenced_specs(g, src_doc)
        for spec_id in sorted(n for n in g.nodes if g.nodes[n].get("kind") == "document"):
            superseders = [
                s for s in _superseding_specs(g, spec_id)
                if _doc_date(g, s) and _doc_date(g, s) <= ev_date
            ]
            if not superseders:
                continue
            if aid not in _assets_of(g, spec_id):
                continue
            if not (set(superseders) & set(used_specs)):
                continue
            newer = max(superseders, key=lambda s: _doc_date(g, s))
            old_b = g.nodes[spec_id].get("body", "")
            new_b = g.nodes[newer].get("body", "")
            old_min = MIN_DIELECTRIC_RE.search(old_b)
            new_min = MIN_DIELECTRIC_RE.search(new_b)
            old_int = BUSING_INTERVAL_RE.search(old_b)
            new_int = BUSING_INTERVAL_RE.search(new_b)
            detail = []
            if old_min and new_min and old_min.group(1) != new_min.group(1):
                detail.append(
                    f"min oil dielectric {old_min.group(1)} kV vs {new_min.group(1)} kV"
                )
            if old_int and new_int and old_int.group(1) != new_int.group(1):
                detail.append(
                    f"bushing interval {old_int.group(1)}y vs {new_int.group(1)}y"
                )
            conflicts.append({
                "rule": "SPEC_VS_EVENT_CONFLICT",
                "node": spec_id,
                "severity": "CONFLICT",
                "message": (
                    f"{spec_id} (v{g.nodes[spec_id].get('version')}) still specifies "
                    f"{'; '.join(detail) or 'different requirements'}, but the latest "
                    f"maintenance event on {aid} ({ev_id}, {ev_date}) was performed "
                    f"under {newer} (v{g.nodes[newer].get('version')}), so the two "
                    f"documents contradict each other."
                ),
            })

    # Rule 3: an event was performed under a specification that was not yet
    # effective on the event's date (in-force violation).
    for ev in events:
        ev_date = g.nodes[ev].get("date") or ""
        if not ev_date:
            continue
        src_doc = _source_doc_of_event(g, ev)
        if not src_doc:
            continue
        for spec_id in _referenced_specs(g, src_doc):
            spec_eff = _doc_date(g, spec_id)
            if spec_eff and spec_eff > ev_date:
                conflicts.append({
                    "rule": "SPEC_NOT_YET_IN_FORCE",
                    "node": ev,
                    "severity": "CONFLICT",
                    "message": (
                        f"{ev} ({ev_date}) was performed under {spec_id} "
                        f"(v{g.nodes[spec_id].get('version')}, eff {spec_eff}), "
                        f"but that specification was not yet effective on the "
                        f"event date, so the work and any results recorded "
                        f"against it are not in force."
                    ),
                })

    # Rule 4: a document references a spec that was already superseded on the
    # document's own date.
    for doc_id in sorted(g.nodes):
        if g.nodes[doc_id].get("kind") != "document":
            continue
        doc_date = _doc_date(g, doc_id)
        if not doc_date:
            continue
        for spec_id in _referenced_specs(g, doc_id):
            for newer in _superseding_specs(g, spec_id):
                if _doc_date(g, newer) and _doc_date(g, newer) <= doc_date:
                    conflicts.append({
                        "rule": "STALE_REFERENCE",
                        "node": doc_id,
                        "severity": "CONFLICT",
                        "message": (
                            f"{doc_id} ({doc_date}) cites {spec_id} as the governing "
                            f"standard, but {newer} was already effective on that date "
                            f"(eff {_doc_date(g, newer)}), so the reference and any "
                            f"conclusions based on it are stale."
                        ),
                    })

    # Rule 5: current asset state in the register vs the governing spec's limits.
    for n, kind in g.nodes(data="kind"):
        if kind != "asset":
            continue
        oil = g.nodes[n].get("oil_dielectric_kv")
        if oil is None:
            continue
        governing = [
            d for d in g.predecessors(n)
            if g.edges[d, n].get("type") == "DOCUMENTS"
            and g.nodes[d].get("doc_type") == "engineering_spec"
            and _doc_date(g, d) and _doc_date(g, d) <= today
        ]
        if not governing:
            continue
        spec_id = max(governing, key=lambda d: _doc_date(g, d))
        m = MIN_DIELECTRIC_RE.search(g.nodes[spec_id].get("body", ""))
        if m and float(oil) < float(m.group(1)):
            conflicts.append({
                "rule": "ASSET_STATE_VS_SPEC",
                "node": n,
                "severity": "CONFLICT",
                "message": (
                    f"{n} is registered with oil dielectric strength {oil} kV "
                    f"(asset register in force), which is below the {m.group(1)} kV "
                    f"minimum of the governing standard {spec_id} "
                    f"(v{g.nodes[spec_id].get('version')}, eff {_doc_date(g, spec_id)})."
                ),
            })

    # Rule 6 (info): documents not yet effective.
    for doc_id in sorted(g.nodes):
        if g.nodes[doc_id].get("kind") != "document":
            continue
        eff = _doc_date(g, doc_id)
        if eff and eff > today:
            conflicts.append({
                "rule": "NOT_YET_EFFECTIVE",
                "node": doc_id,
                "severity": "INFO",
                "message": f"{doc_id} is not effective until {eff}.",
            })

    order = {"CONFLICT": 0, "STALE": 1, "INFO": 2}
    return sorted(conflicts, key=lambda c: (order[c["severity"]], c["node"]))


def report(conflicts):
    """Human-readable conflict report as a list of lines."""
    lines = ["Temporal-validity check", "=" * 24]
    if not conflicts:
        lines.append("No conflicts found.")
        return lines
    for i, c in enumerate(conflicts, 1):
        lines.append(f"{i}. [{c['severity']}] {c['rule']} -> {c['node']}")
        lines.append(f"   {c['message']}")
    lines.append("")
    by_sev = {}
    for c in conflicts:
        by_sev[c["severity"]] = by_sev.get(c["severity"], 0) + 1
    lines.append("Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(by_sev.items())))
    return lines
