"""Offline acceptance of the actual Deep Research path with deterministic evidence."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gpt_researcher.agent import GPTResearcher
from gpt_researcher.skills.adaptive_research import AdaptiveController, redact
from gpt_researcher.skills.curator import SourceCurator, qualify_finding
from gpt_researcher.skills.deep_research import DeepResearchSkill
from gpt_researcher.skills.knowledge_export import export_knowledge, validate_export, write_artifacts
from gpt_researcher.skills.research_observer import ResearchLimitReached, research_observer
from gpt_researcher.skills.research_state import (Contradiction, Evaluation, Finding, Hypothesis,
    Investigation, Question, ResearchState, StateStore, normalize_query)


QUOTE = "Participants reported choosing coffee for convenience."
COUNTER = "I stopped buying this coffee because it takes too long to dilute."
STUDY = {"url": "https://study.example/original", "title": "Original study",
         "raw_content": "An original exploratory interview study of 24 U.S. coffee drinkers in 2025. " + QUOTE}
REVIEW = {"url": "https://reviews.example/account", "title": "Individual product review", "raw_content": COUNTER}


def parent(tmp_path, **overrides):
    cfg = SimpleNamespace(deep_research_breadth=1, deep_research_depth=2,
        deep_research_concurrency=4, deep_research_max_waves=2,
        deep_research_max_seconds=20., deep_research_max_cost_usd=5.,
        deep_research_max_searches=12, deep_research_min_gain=.15, deep_research_call_timeout=2.,
        strategic_llm_provider="openai", strategic_llm_model="fixture-model", smart_llm_model="fixture-model",
        strategic_token_limit=32000, smart_token_limit=32000, llm_kwargs={}, reasoning_effort="medium",
        max_search_results_per_query=5, research_state_dir=str(tmp_path), config_path=None)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return SimpleNamespace(cfg=cfg, websocket=None, tone=None, headers={}, visited_urls=set(),
        query="Understand coffee convenience and preparation trade-offs", query_domains=[],
        retrievers=[type("FixtureRetriever", (), {})], mcp_configs=None, mcp_strategy=None,
        add_costs=lambda _: None, get_costs=lambda: 0., log_handler=None)


def assessment(evidence):
    primary = evidence.id == "S001"
    return {"id": evidence.id, "kind": "primary_research" if primary else "firsthand_consumer",
        "independence_group": "study-24" if primary else "account-1", "independence_known": True,
        "publisher": "Fixture study" if primary else "Fixture reviewer", "publication_date": None,
        "population": "24 U.S. interview participants" if primary else None,
        "geography": ["United States"] if primary else [], "period": "2025" if primary else None,
        "methodology": "exploratory interviews, convenience sample" if primary else None,
        "measure": "self_reported", "suitability": ["reported_experience", "hypothesis_discovery"],
        "quotes": [QUOTE if primary else COUNTER], "credibility": "Traceable firsthand material; directional only.",
        "limitations": ["Not representative; cannot establish prevalence."]}


def evaluation(wave):
    fact = Finding(id="F001", text=QUOTE, evidence_ids=["S001"], question_ids=["Q001"],
                   scope="24 interviewed U.S. coffee drinkers in 2025")
    return Evaluation(findings=[fact],
        questions=[Question(id="Q001", question="Do consumers value convenience?", acceptance_criterion="Direct bounded evidence and competing preparation explanations",
            status="Partial", evidence_ids=["S001"] if wave == 1 else ["S001", "S002"], gap="Representativeness and repeat purchase remain unknown")],
        hypotheses=[Hypothesis(id="H001", text="Preparation burden may offset convenience", status="supported",
            supporting_evidence_ids=["S001"], opposing_evidence_ids=[] if wave == 1 else ["S002"],
            alternatives=["Different formats and usage occasions"], rationale="Compare observed contexts")],
        contradictions=[] if wave == 1 else [Contradiction(id="C001", description="Convenience benefit contrasts with preparation burden in another context",
            evidence_ids=["S001", "S002"], question_ids=["Q001"], possible_explanations=["Format differences; self-selection"])],
        candidates=[Investigation(action="CHALLENGE", query="negative firsthand preparation accounts", question_ids=["Q001"], evidence_ids=["S001"], hypothesis_ids=["H001"],
            expected_gain=.9, rationale="Test the convenience explanation against preparation burden"),
            Investigation(action="DEEPEN", query="more convenience coverage", question_ids=["Q001"], evidence_ids=["S001"],
            expected_gain=.3, rationale="Less discriminating repetition of known context")] if wave == 1 else [],
        stop_requested=wave > 1, stopping_rationale="Partial evidence; investigate only if new routes are worthwhile")


def install_fixtures(monkeypatch):
    queries = []
    async def search(query, retriever, **kwargs):
        queries.append(query)
        return [{"href": REVIEW["url"] if "negative" in query else STUDY["url"], "body": "Search snippet is not source evidence"}]
    class Child:
        def __init__(self, **kwargs):
            self.cfg = SimpleNamespace()
            self.source_urls = []
            self.visited_urls = set()
            self.research_sources = []
        async def conduct_research(self):
            self.research_sources = [copy.deepcopy(STUDY if u == STUDY["url"] else REVIEW) for u in self.source_urls]
            self.visited_urls.update(self.source_urls)
            return [s["raw_content"] for s in self.research_sources]
    async def generate(self, query, num_queries=3):
        return [{"query": "initial convenience study", "researchGoal": "Do consumers value convenience and encounter preparation trade-offs?"}]
    async def ask(self, prompt, purpose):
        if purpose == "evidence_qualification":
            return json.dumps({"assessments": [assessment(e) for e in self.state.evidence.values() if not e.assessed]})
        assert "Accumulated state" in prompt
        assert "S001" in prompt
        return evaluation(self.state.wave).model_dump_json()
    monkeypatch.setattr("gpt_researcher.GPTResearcher", Child)
    monkeypatch.setattr("gpt_researcher.actions.query_processing.get_search_results", search)
    monkeypatch.setattr(DeepResearchSkill, "generate_search_queries", generate)
    monkeypatch.setattr(AdaptiveController, "ask", ask)
    return queries


@pytest.mark.asyncio
async def test_actual_deep_path_selects_evidence_driven_challenge_and_exports(tmp_path, monkeypatch):
    queries = install_fixtures(monkeypatch)
    p = parent(tmp_path)
    skill = DeepResearchSkill(p)
    context = await skill.run()
    assert queries == ["initial convenience study", "negative firsthand preparation accounts"]
    assert skill.state.wave == 2
    assert "S001" in context and "C001" in context
    assert skill.state.hypotheses["H001"].status == "contested"
    assert skill.state.findings["F001"].confidence == "Verified"
    assert skill.state.status == "partial"
    assert any(e["event"] == "premature_stop_rejected" for e in skill.state.events)
    decisions = [e for e in skill.state.events if e["event"] == "decision"]
    assert decisions and decisions[0]["data"]["accumulated_evidence_ids"] == ["S001"]
    assert any(e["data"]["investigation"]["status"] == "deferred" for e in decisions)
    payload = export_knowledge(skill.state)
    validate_export(payload)
    assert payload["schema_version"] == "1.2"
    assert payload["completion_review"]["answer_status"] == "Partial"
    assert all(r["properties"]["Relationships"] == [] for r in payload["knowledge_records"])
    assert "Confidence" in payload["knowledge_records"][0]["properties"]
    assert "Credibility" not in payload["knowledge_records"][0]["properties"]
    files = write_artifacts(skill.state, tmp_path / "outputs")
    assert json.loads(Path(files["trace"]).read_text())["events"]
    assert Path(files["md"]).read_text() == payload["report_markdown"]
    assert StateStore(tmp_path).load(skill.state.run_id).model_dump() == skill.state.model_dump()


@pytest.mark.asyncio
async def test_completed_investigation_is_not_repeated_after_interruption(tmp_path, monkeypatch):
    queries = install_fixtures(monkeypatch)
    original = AdaptiveController.evaluate
    async def interrupt(self):
        raise asyncio.CancelledError()
    monkeypatch.setattr(AdaptiveController, "evaluate", interrupt)
    skill = DeepResearchSkill(parent(tmp_path))
    with pytest.raises(asyncio.CancelledError):
        await skill.run()
    assert skill.state.status == "interrupted"
    assert skill.state.investigations[0].status == "completed"
    monkeypatch.setattr(AdaptiveController, "evaluate", original)
    resumed = DeepResearchSkill(parent(tmp_path))
    await resumed.run()
    assert resumed.state.run_id == skill.state.run_id
    assert queries.count("initial convenience study") == 1
    assert any(e["event"] == "resumed" for e in resumed.state.events)


def opened_state():
    s = ResearchState(fingerprint="fixture", query="Consumer test")
    e = s.add_source(STUDY)
    e.assessed = True
    e.kind = "primary_research"
    e.quotes = [QUOTE]
    e.population, e.geography, e.period, e.methodology = "24 interviewees", ["United States"], "2025", "convenience interviews"
    return s


@pytest.mark.parametrize("kind,claim_type", [("interpretation", "bounded_fact"), ("hypothesis", "bounded_fact"), ("finding", "prevalence"), ("finding", "causal")])
def test_unsupported_inference_never_promoted_to_verified(kind, claim_type):
    s = opened_state()
    f = Finding(id="F1", text="Most U.S. consumers buy coffee because convenience causes repeat purchase", kind=kind,
        claim_type=claim_type, evidence_ids=["S001"], confidence="Verified", scope="All U.S. consumers")
    assert qualify_finding(f, s.evidence).confidence != "Verified"


def test_uncited_claim_is_unverified_and_review_cannot_establish_prevalence():
    s = opened_state()
    s.evidence["S001"].kind = "firsthand_consumer"
    f = Finding(id="F1", text=QUOTE, evidence_ids=["S001"], scope="Individual review")
    assert qualify_finding(f, s.evidence).confidence == "Probable"
    f.evidence_ids = []
    assert qualify_finding(f, s.evidence).confidence == "Unverified"


@pytest.mark.asyncio
async def test_qualification_rejects_invented_quotes_and_keeps_unknown(tmp_path):
    s = opened_state()
    s.evidence["S001"].assessed = False
    a = assessment(s.evidence["S001"])
    a["quotes"] = ["This quote was never in the source"]
    await SourceCurator(parent(tmp_path)).qualify_evidence(s, ["S001"], AsyncMock(return_value=json.dumps({"assessments": [a]})))
    assert s.evidence["S001"].kind == "unknown"
    assert not s.evidence["S001"].quotes
    assert not s.evidence["S001"].suitability


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["firsthand_consumer", "primary_research", "secondary_reporting", "opinion"])
async def test_classifies_evidence_kinds_without_assuming_authority_is_support(tmp_path, kind):
    s = opened_state()
    a = assessment(s.evidence["S001"])
    a["kind"] = kind
    a["suitability"] = ["population_prevalence", "causal_inference", "reported_experience"]
    await SourceCurator(parent(tmp_path)).qualify_evidence(s, ["S001"], AsyncMock(return_value=json.dumps({"assessments": [a]})))
    assert s.evidence["S001"].kind == kind
    if kind != "primary_research":
        assert "population_prevalence" not in s.evidence["S001"].suitability


def test_state_roundtrip_and_duplicate_run_lock(tmp_path):
    store = StateStore(tmp_path)
    state = store.open("same assignment", {"breadth": 3})
    state.hypotheses["H1"] = Hypothesis(id="H1", text="A testable alternative")
    store.save(state)
    with pytest.raises(RuntimeError, match="already running"):
        StateStore(tmp_path).open("same assignment", {"breadth": 3})
    store.close()
    second = StateStore(tmp_path)
    loaded = second.open("same assignment", {"breadth": 3})
    assert loaded.run_id == state.run_id and "H1" in loaded.hypotheses
    second.close()


def test_query_and_source_deduplication():
    s = opened_state()
    assert s.add_source({**STUDY, "url": STUDY["url"] + "?utm_source=repeat#fragment"}).id == "S001"
    assert len(s.evidence) == 1
    assert s.add_source({"url": "https://a.example", "body": "Only a search snippet"}) is None
    assert normalize_query("Repeat! Search") == normalize_query("repeat search")


def test_select_rejects_redundancy_unknown_refs_and_unjustified_actions(tmp_path):
    c = AdaptiveController(DeepResearchSkill(parent(tmp_path)), 3, 2)
    c.state = opened_state()
    c.state.questions["Q1"] = Question(id="Q1", question="why", acceptance_criterion="Direct accounts")
    c.state.investigations = [Investigation(query="Repeated QUERY", status="completed")]
    result = c.select(Evaluation(candidates=[
        Investigation(query="repeated query", question_ids=["Q1"], rationale="repeat"),
        Investigation(query="new", question_ids=["missing"], rationale="missing"),
        Investigation(query="low value", question_ids=["Q1"], rationale="low", expected_gain=.01)]))
    assert result == []
    assert all(x.status == "rejected" for x in c.state.investigations[1:])


@pytest.mark.asyncio
async def test_failed_search_and_malformed_evaluation_produce_partial_state(tmp_path, monkeypatch):
    queries = install_fixtures(monkeypatch)
    async def bad_search(*a, **k):
        raise RuntimeError("Search inaccessible")
    monkeypatch.setattr("gpt_researcher.actions.query_processing.get_search_results", bad_search)
    skill = DeepResearchSkill(parent(tmp_path))
    await skill.run()
    assert skill.state.status == "partial" and not skill.state.findings
    assert any(e["event"] == "investigation_failed" for e in skill.state.events)
    validate_export(export_knowledge(skill.state))


@pytest.mark.asyncio
async def test_malformed_evaluation_does_not_trigger_duplicate_searches(tmp_path, monkeypatch):
    queries = install_fixtures(monkeypatch)
    original = AdaptiveController.ask
    async def bad(self, prompt, purpose):
        if purpose == "evaluate_accumulated_evidence":
            return "not an evaluation"
        return await original(self, prompt, purpose)
    monkeypatch.setattr(AdaptiveController, "ask", bad)
    skill = DeepResearchSkill(parent(tmp_path))
    await skill.run()
    assert len(queries) == 1 and skill.state.status == "partial" and not skill.state.findings


def test_limits_are_checked_before_dispatch(tmp_path):
    c = AdaptiveController(DeepResearchSkill(parent(tmp_path)), 3, 2)
    c.state = opened_state()
    c.state.known_cost_usd = 6
    with pytest.raises(ResearchLimitReached, match="cost"):
        c.before_call()


def test_model_observation_captures_usage_and_redacts_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "private-fixture-secret-do-not-export")
    c = AdaptiveController(DeepResearchSkill(parent(tmp_path)), 3, 2)
    c.state = opened_state()
    p = SimpleNamespace(last_usage_metadata={"input_tokens": 10, "output_tokens": 4}, last_response_metadata={"finish_reason": "stop"})
    c.record_call([{"role": "user", "content": "private-fixture-secret-do-not-export"}], "fixture", {}, "answer", p, 1, None, .01)
    assert c.state.known_cost_usd == .01 and c.state.input_tokens == 10
    assert "private-fixture-secret-do-not-export" not in json.dumps(c.state.events)
    assert c.state.events[-1]["data"]["messages"][0]["content"] == "[REDACTED]"


def test_export_validation_rejects_fake_notion_relationship_and_broken_source():
    payload = export_knowledge(opened_state())
    bad = copy.deepcopy(payload)
    bad["knowledge_records"][0]["properties"]["Relationships"] = ["K001"]
    with pytest.raises(ValueError): validate_export(bad)
    bad = copy.deepcopy(payload)
    bad["knowledge_records"][0]["source_ids"] = ["invented"]
    with pytest.raises(ValueError): validate_export(bad)


@pytest.mark.asyncio
async def test_agent_writer_publishes_state_not_another_ungrounded_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = opened_state()
    fake = SimpleNamespace(deep_researcher=SimpleNamespace(state=s), websocket=None)
    report = await GPTResearcher.write_report(fake)
    assert "## Unanswered questions" in report
    assert Path(fake.research_artifacts["json"]).is_file()
    assert Path(fake.research_artifacts["trace"]).is_file()
