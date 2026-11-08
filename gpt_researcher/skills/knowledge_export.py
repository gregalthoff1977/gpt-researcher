"""One validated state produces both the report and the exact v1.1 mapping."""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import jsonschema

from .curator import qualify_finding
from .research_state import ResearchState

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def _citations(state, ids):
    return " ".join(f"[{sid}]({state.evidence[sid].url})" for sid in ids if sid in state.evidence)


def render_markdown(state: ResearchState) -> str:
    lines = ["# Consumer research report", "", "## Scope and method", "", state.query,
             "", f"Run `{state.run_id}`. Research status: **{state.status}**. "
             "This desk research distinguishes opened source evidence, consumer self-reports, and analyst interpretation. "
             "Online accounts are self-selected and cannot establish population prevalence.",
             "", "## Verified and bounded findings", ""]
    facts = [f for f in state.findings.values() if f.kind == "finding"]
    for f in facts:
        lines.extend([f"### {f.id} — {f.confidence}", "", f"{f.text} {_citations(state, f.evidence_ids)}",
                      "", f"Scope: {f.scope or 'Not fully established.'} {f.rationale}", ""])
    if not facts:
        lines.append("No claim-qualified factual conclusions were established. Retained source material is not itself a verified finding.")
    lines += ["", "## Supported interpretations and consumer insights", ""]
    insights = [f for f in state.findings.values() if f.kind == "interpretation" and f.confidence == "Probable"]
    for f in insights:
        lines += [f"### {f.id} — {f.confidence}", "", f"{f.text} {_citations(state, f.evidence_ids)}",
                  "", "Supporting findings: " + (", ".join(f.supporting_finding_ids) or "None; interpretation remains a hypothesis."),
                  "", "Competing explanations: " + ("; ".join(f.alternatives) or "Not established."),
                  "", "Validation needed: " + (f.validation_test or "Not specified; do not treat as established."), ""]
    if not insights:
        lines.append("No defensible consumer insight was established beyond the retained findings.")
    lines += ["", "## Unverified hypotheses", ""]
    for h in state.hypotheses.values():
        lines += [f"- **{h.id} ({h.status})**: {h.text} {_citations(state, h.supporting_evidence_ids + h.opposing_evidence_ids)} "
                  f"{h.rationale} Alternatives: {'; '.join(h.alternatives) or 'Unknown.'}"]
    for f in state.findings.values():
        if f.kind == "hypothesis" or (f.kind == "interpretation" and f.confidence != "Probable"):
            lines += [f"- **{f.id} ({f.confidence})**: {f.text} {_citations(state, f.evidence_ids)} "
                      f"Alternatives: {'; '.join(f.alternatives) or 'Unestablished'}. Validation: {f.validation_test or 'Not specified.'}"]
    if not state.hypotheses and not any(f.kind == "hypothesis" for f in state.findings.values()):
        lines.append("No separate testable hypothesis was retained.")
    lines += ["", "## Contradictions and competing evidence", ""]
    for c in state.contradictions.values():
        lines += [f"- **{c.id}**: {c.description} {_citations(state, c.evidence_ids)} "
                  f"Possible explanations: {'; '.join(c.possible_explanations) or 'Unresolved.'}"]
    if not state.contradictions:
        lines.append("No explicit contradiction was retained. This does not establish that the available evidence is unanimous.")
    lines += ["", "## Unanswered questions and coverage", ""]
    for q in state.questions.values():
        lines += [f"- **{q.id} [{q.importance}; {q.status}]** {q.question} {_citations(state, q.evidence_ids)} "
                  f"Acceptance criterion: {q.acceptance_criterion} Remaining gap: {q.gap}"]
    lines += ["", "## Stopping decision and limitations", "", state.stopping_reason or "Research was interrupted.",
              "", f"Executed search actions: {state.search_actions}. Elapsed research time: {state.elapsed_seconds:.1f} seconds. "
              f"Available model cost subtotal: ${state.known_cost_usd:.4f}. "
              "Search-provider fees and any unpriced model calls are not included; this is not a reconciled billing total.", ""]
    lines += [f"- {x}" for x in dict.fromkeys(state.limitations)]
    lines += ["- Evidence classification is automated. Claim support and research quality remain subject to human review.",
              "- Unknown population, methods, dates, and independence remain unknown; no synthetic consumer testimony was generated.",
              "", "## Sources and evidence qualification", ""]
    for e in state.evidence.values():
        lines += [f"### [{e.id}: {e.title}]({e.url})", "", f"Evidence type: {e.kind}. Measure: {e.measure}. "
                  f"Population: {e.population or 'Unknown'}. Geography: {', '.join(e.geography) or 'Unknown'}. "
                  f"Data period: {e.period or 'Unknown'}. Method: {e.methodology or 'Unknown'}. "
                  f"Publication date: {e.publication_date or 'Unknown'}. Accessed: {e.accessed_at[:10]}.",
                  "", f"Claim suitability: {', '.join(e.suitability) or 'Unqualified'}. {e.credibility}",
                  "", "Limitations: " + ("; ".join(e.limitations) or "No additional limitation documented; not proof of validity."), ""]
    return "\n".join(lines).strip() + "\n"


def _record(kid, name, kind, text, evidence, confidence, rationale, *, topics=None, entities=None, geography=None, implications=""):
    return {"id": kid, "properties": {
        "Name": name[:180], "Type": kind, "Content": text,
        "Source URL": evidence[0].url if evidence else None,
        "Evidence": "\n".join(f"[{e.id}] {e.url} — {e.credibility} Population: {e.population or 'unknown'}; method: {e.methodology or 'unknown'}; period: {e.period or 'unknown'}. Limitations: {'; '.join(e.limitations)}" for e in evidence),
        "Topics": list(dict.fromkeys(topics or [])), "Entities": list(dict.fromkeys(entities or [])),
        "Relationships": [], "Confidence": confidence, "Geography": list(dict.fromkeys(geography or [])),
        "Time Period": None, "Implications": implications, "Status": "Captured"},
        "source_ids": [e.id for e in evidence], "confidence_rationale": rationale,
        "metadata_notes": ["Existing Notion page UUIDs were not checked; local links are in proposed_relationships.",
                           "Exact data-period start/end dates were not established; no day was invented."]}


def export_knowledge(state: ResearchState) -> dict:
    """The authoritative Consumer Research Director v1.1 envelope (schema 1.2)."""
    records, edges, audit = [], [], []
    source_records, finding_records = {}, {}
    for e in state.evidence.values():
        kid = f"K{len(records) + 1:03d}"
        source_records[e.id] = kid
        records.append(_record(kid, e.title, "Source", f"Opened source: {e.title}. Evidence class: {e.kind}. {e.credibility}", [e],
                               "Probable" if e.assessed and e.quotes else "Unverified", "Source retained as reference; publisher credibility is not proof of any broader consumer claim.", geography=e.geography))
    for f in state.findings.values():
        qualify_finding(f, state.evidence, state.findings)
        evidence = [state.evidence[sid] for sid in f.evidence_ids if sid in state.evidence]
        kid = f"K{len(records) + 1:03d}"
        finding_records[f.id] = kid
        text = f.text + (" Scope: " + f.scope if f.scope else "")
        if f.alternatives:
            text += " Competing explanations: " + "; ".join(f.alternatives)
        if f.validation_test:
            text += " Validation needed: " + f.validation_test
        records.append(_record(kid, f.text, {"finding": "Finding", "interpretation": "Insight", "hypothesis": "Observation"}[f.kind],
                               text, evidence, f.confidence, f.rationale, topics=f.topics, entities=f.entities,
                               geography=f.geography, implications=f.implications))
        for e in evidence:
            edges.append({"from_id": source_records[e.id], "to_id": kid, "relationship": "supports" if f.confidence in ("Verified", "Probable") else "contextualizes",
                          "rationale": f"Qualified source context for {f.id}; strength and limitations remain in the claim audit."})
        checks = [{"source_id": e.id, "url": e.url, "location": "Captured opened excerpt; source text hash retained in research state",
                   "supporting_detail": " | ".join(e.quotes), "access_level": "Secondary summary" if e.kind == "secondary_reporting" else "Full text",
                   "verdict": "Supported" if f.confidence == "Verified" else "Partially supported" if e.quotes else "Unassessable",
                   "scope_notes": f.scope + "; " + "; ".join(e.limitations)} for e in evidence]
        audit.append({"claim_id": f.id, "record_ids": [kid], "claim": f.text, "material": True, "checks": checks,
                      "overall_verdict": "Supported" if f.confidence == "Verified" else "Partially supported" if evidence else "Unassessable",
                      "correction_or_qualification": f.rationale})
    for f in state.findings.values():
        for parent in f.supporting_finding_ids:
            if parent in finding_records:
                edges.append({"from_id": finding_records[parent], "to_id": finding_records[f.id],
                              "relationship": "suggests", "rationale": "Evidence-grounded interpretation; not a promotion to verified fact."})
    for h in state.hypotheses.values():
        evidence = [state.evidence[sid] for sid in dict.fromkeys(h.supporting_evidence_ids + h.opposing_evidence_ids)]
        kid = f"K{len(records) + 1:03d}"
        records.append(_record(kid, h.text, "Observation", f"Hypothesis ({h.status}): {h.text} {h.rationale} Competing explanations: {'; '.join(h.alternatives)}", evidence, "Hypothesis", "Testable interpretation; retained supporting and opposing evidence is not verification."))
    for c in state.contradictions.values():
        for left in c.evidence_ids[:1]:
            for right in c.evidence_ids[1:]:
                edges.append({"from_id": source_records[left], "to_id": source_records[right], "relationship": "contradicts", "rationale": c.description})
    sources = [{"id": e.id, "title": e.title, "url": e.url, "publisher": e.publisher,
        "publication_date": e.publication_date, "accessed_date": e.accessed_at[:10],
        "source_kind": "Primary" if e.kind in ("primary_research", "firsthand_consumer", "official_disclosure") else "Secondary" if e.kind == "secondary_reporting" else "Other",
        "independence_group": e.independence_group, "credibility_notes": e.credibility,
        "limitations": e.limitations + ([] if e.independence_known else ["Source independence is not established."])} for e in state.evidence.values()]
    schema_snapshot = json.loads((SCHEMA_DIR / "notion_schema_snapshot.json").read_text())
    vocabulary = {field: {o["name"] for o in schema_snapshot["schema"][field].get("options", [])} for field in ("Topics", "Entities", "Geography")}
    proposed = {field: sorted({v for r in records for v in r["properties"][field] if v not in vocabulary[field]}) for field in vocabulary}
    coverage = []
    for q in state.questions.values():
        tasks = [i for i in state.investigations if q.id in i.question_ids]
        status = q.status
        refs = [sid for sid in q.evidence_ids if sid in state.evidence]
        attempts = [{"purpose": i.action + ": " + i.rationale, "query_or_url": i.query,
                     "observed_result": i.status + ": " + i.result} for i in tasks if i.attempts]
        requirement = {"id": q.id, "requirement": q.question, "importance": q.importance,
            "acceptance_criterion": q.acceptance_criterion, "answer_status": status,
            "answer_summary": " ".join(f.text for f in state.findings.values() if q.id in f.question_ids),
            "source_ids": refs, "remaining_gap": q.gap,
            "disposition": "Answered" if status == "Sufficient" else "Effort limited"}
        coverage.append({"id": q.id, "subquestion": q.question, "importance": q.importance,
            "evidence_status": "Mixed" if refs else "Unresolved", "source_ids": refs,
            "investigation_attempts": attempts, "remaining_gap": q.gap, "completion_basis": state.stopping_reason,
            "answer_status": status, "answer_requirements": [requirement]})
    critical = [q for q in state.questions.values() if q.importance == "Critical"]
    partial = [q.id for q in critical if q.status == "Partial"]
    unresolved = [q.id for q in critical if q.status == "Unresolved"]
    warnings = list(dict.fromkeys(state.limitations))
    for field, values in proposed.items():
        if values:
            warnings.append(f"Proposed {field} options require acceptance before any future Notion insertion: {', '.join(values)}")
    warnings.extend(["No live Notion schema refresh or cross-run Notion deduplication was performed.",
                     "Freshness completeness was not separately verified; completion review remains partial even if substantive questions are answered.",
                     "Confidence gates check provenance and scope; automated classification is not an independent factual audit.",
                     "Available costs are subtotals, not a complete bill."])
    complete = state.status == "complete" and not partial and not unresolved
    result = {"schema_version": "1.2", "research_objective": state.query, "as_of_date": state.updated_at[:10],
        "parameters": {"depth": "Deep", "breadth": "Evidence-driven investigations within configured breadth/depth/wave limits",
            "verification": "Claim-specific source qualification and conservative deterministic confidence gates",
            "insight_emphasis": "Consumer behavior, motivations, tensions, trade-offs and competing explanations",
            "extraction_granularity": "One reusable bounded claim or explicitly labeled interpretation per record"},
        "research_log": {"plan": [q.question for q in state.questions.values()],
            "investigations": [f"{i.action}: {i.query} — {i.status}: {i.result}" for i in state.investigations if i.attempts],
            "follow_up_investigations": [f"{i.action}: {i.query} — {i.rationale}" for i in state.investigations if i.action != "INVESTIGATE" and i.attempts],
            "verification_checks": ["Checked retained quote anchors against opened excerpts", "Applied claim-level confidence gates", "Validated exact Knowledge property names and local relationship references"],
            "stopping_reason": state.stopping_reason, "limitations": warnings},
        "report_markdown": render_markdown(state), "sources": sources, "knowledge_records": records,
        "proposed_relationships": edges, "proposed_options": proposed,
        "validation": {"status": "passed" if complete else "partial", "checks": ["JSON Schema contract validation", "Source/record/relationship referential integrity", "No unchecked Notion relations", "Verified findings constrained by deterministic evidence gates"],
            "errors": [], "warnings": warnings, "notion_write_performed": False},
        "usage": {"search_actions": state.search_actions, "input_tokens": state.input_tokens or None,
            "output_tokens": state.output_tokens or None, "estimated_cost_usd": state.known_cost_usd if state.cost_complete else None,
            "cost_notes": f"Available model cost subtotal ${state.known_cost_usd:.6f}; completeness={state.cost_complete}. Search-provider fees and unpriced/in-flight calls are not included."},
        "coverage": coverage, "claim_audit": audit, "opportunity_assessments": [],
        "freshness_checks": [{"id": "FR" + q.id, "requirement_ids": [q.id], "result": "Not checked", "attempts": [],
            "retained_source_ids": q.evidence_ids, "retained_data_period": "Unknown unless stated in the source registry",
            "conclusion": "No separate latest-evidence completeness claim is made."} for q in critical],
        "completion_review": {"answer_status": "Partial" if records else "Unresolved",
            "critical_partial_requirement_ids": partial, "critical_unresolved_requirement_ids": unresolved,
            "incomplete_freshness_requirement_ids": [q.id for q in critical],
            "stopping_basis": "Adequate evidence" if complete else "Diminishing returns" if "Diminishing" in state.stopping_reason else "Effort limit",
            "reason": state.stopping_reason}}
    validate_export(result)
    return result


def validate_export(payload):
    schema = json.loads((SCHEMA_DIR / "consumer_research_director_v1_1.json").read_text())["schema"]
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(payload)
    records = {r["id"]: r for r in payload["knowledge_records"]}
    sources = {s["id"]: s for s in payload["sources"]}
    if len(records) != len(payload["knowledge_records"]) or len(sources) != len(payload["sources"]):
        raise ValueError("Duplicate stable record/source ID")
    from .research_state import canonical_url
    for source in sources.values():
        canonical_url(source["url"])
    for record in records.values():
        if not set(record["source_ids"]) <= sources.keys():
            raise ValueError("Unresolved source reference")
        props = record["properties"]
        if not props["Name"].strip() or not props["Content"].strip() or props["Relationships"]:
            raise ValueError("Empty record or unverified Notion relationship")
        if props["Source URL"] and props["Source URL"] not in {sources[s]["url"] for s in record["source_ids"]}:
            raise ValueError("Source URL does not resolve to supporting registry")
        if props["Confidence"] == "Verified":
            audits = [a for a in payload["claim_audit"] if record["id"] in a["record_ids"]]
            if props["Type"] != "Finding" or not audits or any(a["overall_verdict"] != "Supported" for a in audits):
                raise ValueError("Unsupported promotion to Verified")
    for edge in payload["proposed_relationships"]:
        if edge["from_id"] not in records or edge["to_id"] not in records:
            raise ValueError("Unresolved local relationship")
    if payload["validation"]["notion_write_performed"]:
        raise ValueError("Notion writes are prohibited")


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".export-")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_artifacts(state, output_dir="outputs"):
    from .adaptive_research import redact
    payload = export_knowledge(state)
    state.record("outputs_validated", knowledge_records=len(payload["knowledge_records"]), source_count=len(payload["sources"]),
                 contract="Consumer Research Director v1.1 / schema 1.2", notion_write_performed=False)
    if state._checkpoint_directory:
        from .research_state import StateStore
        StateStore(state._checkpoint_directory).save(state)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "research_" + state.run_id
    knowledge = directory / f"{stem}.knowledge.json"
    trace = directory / f"{stem}.trace.json"
    snapshot = directory / f"{stem}.state.json"
    markdown = directory / f"{stem}.md"
    atomic_json(knowledge, redact(payload))
    atomic_json(trace, redact({"schema_version": "adaptive-trace-1", "run_id": state.run_id,
        "status": state.status, "events": state.events, "elapsed_seconds": state.elapsed_seconds,
        "search_actions": state.search_actions, "available_cost_usd": state.known_cost_usd,
        "cost_complete": state.cost_complete, "stopping_reason": state.stopping_reason}))
    atomic_json(snapshot, redact(state.model_dump(mode="json")))
    markdown.write_text(payload["report_markdown"], encoding="utf-8")
    return {"json": str(knowledge), "knowledge": str(knowledge), "trace": str(trace), "state": str(snapshot), "md": str(markdown)}
