"""Validated research contracts and single-replica durable checkpoints.

SQLite transactions checkpoint completed work; an OS lock prevents two requests
from researching the same active assignment. No external database is required.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"\w+", query.casefold()))


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Evidence requires a public HTTP(S) source URL without credentials")
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_") and k not in ("fbclid", "gclid")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", urlencode(sorted(query)), ""))


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Question(Contract):
    id: str
    question: str = Field(min_length=1)
    acceptance_criterion: str = Field(min_length=1)
    importance: Literal["Critical", "Supporting"] = "Critical"
    status: Literal["Unresolved", "Partial", "Sufficient"] = "Unresolved"
    evidence_ids: list[str] = Field(default_factory=list)
    gap: str = "Not yet investigated"


class Evidence(Contract):
    id: str
    url: str
    title: str
    text: str
    content_hash: str
    accessed_at: str = Field(default_factory=utcnow)
    assessed: bool = False
    kind: Literal["firsthand_consumer", "primary_research", "official_disclosure", "secondary_reporting", "opinion", "unknown"] = "unknown"
    independence_group: str = "unknown"
    independence_known: bool = False
    publisher: str | None = None
    publication_date: str | None = None
    population: str | None = None
    geography: list[str] = Field(default_factory=list)
    period: str | None = None
    methodology: str | None = None
    measure: Literal["observed", "self_reported", "interpretation", "unknown"] = "unknown"
    suitability: list[Literal["attributed_statement", "reported_experience", "population_prevalence", "causal_inference", "hypothesis_discovery"]] = Field(default_factory=list)
    quotes: list[str] = Field(default_factory=list)
    credibility: str = "Unassessed; credibility alone does not establish claim support."
    limitations: list[str] = Field(default_factory=list)


class Finding(Contract):
    id: str
    text: str = Field(min_length=1)
    kind: Literal["finding", "interpretation", "hypothesis"] = "finding"
    claim_type: Literal["bounded_fact", "prevalence", "causal", "consumer_insight"] = "bounded_fact"
    evidence_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    supporting_finding_ids: list[str] = Field(default_factory=list)
    confidence: Literal["Verified", "Probable", "Hypothesis", "Unverified"] = "Unverified"
    rationale: str = "Evidence has not been checked."
    scope: str = ""
    alternatives: list[str] = Field(default_factory=list)
    validation_test: str = ""
    implications: str = ""
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    geography: list[str] = Field(default_factory=list)


class Hypothesis(Contract):
    id: str
    text: str
    status: Literal["untested", "supported", "weakened", "contested"] = "untested"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    opposing_evidence_ids: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    rationale: str = ""


class Contradiction(Contract):
    id: str
    description: str
    evidence_ids: list[str] = Field(min_length=2)
    question_ids: list[str] = Field(default_factory=list)
    possible_explanations: list[str] = Field(default_factory=list)
    resolved: bool = False


Action = Literal["INVESTIGATE", "DEEPEN", "CHALLENGE", "REDIRECT", "STOP"]


class Investigation(Contract):
    id: str = Field(default_factory=lambda: "I" + uuid.uuid4().hex[:12])
    action: Action = "INVESTIGATE"
    query: str = ""
    question_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    expected_gain: float = Field(default=0.5, ge=0, le=1)
    importance: float = Field(default=1, ge=0, le=1)
    effort: float = Field(default=1, gt=0, le=10)
    status: Literal["pending", "running", "completed", "failed", "deferred", "rejected"] = "pending"
    result: str = ""
    attempts: int = 0
    cost_usd: float | None = None
    cost_complete: bool = False


class Evaluation(Contract):
    findings: list[Finding] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    candidates: list[Investigation] = Field(default_factory=list)
    stop_requested: bool = False
    stopping_rationale: str = ""


class ResearchState(Contract):
    _checkpoint_directory: str = PrivateAttr(default="")
    version: str = "adaptive-consumer-v1"
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    fingerprint: str
    query: str
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)
    status: Literal["running", "interrupted", "complete", "partial"] = "running"
    wave: int = 0
    evaluated_wave: int = 0
    questions: dict[str, Question] = Field(default_factory=dict)
    evidence: dict[str, Evidence] = Field(default_factory=dict)
    findings: dict[str, Finding] = Field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = Field(default_factory=dict)
    contradictions: dict[str, Contradiction] = Field(default_factory=dict)
    investigations: list[Investigation] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    events: list[dict] = Field(default_factory=list)
    stopping_reason: str = ""
    limitations: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0
    known_cost_usd: float = 0
    cost_complete: bool = True
    search_actions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def record(self, event: str, **data) -> None:
        self.updated_at = utcnow()
        self.events.append({"sequence": len(self.events) + 1, "timestamp": self.updated_at,
                            "run_id": self.run_id, "wave": self.wave, "event": event, "data": data})

    def add_source(self, source: dict) -> Evidence | None:
        raw = source.get("raw_content") or source.get("content") or source.get("Content")
        url = source.get("url") or source.get("Source")
        if not isinstance(raw, str) or not raw.strip() or not isinstance(url, str):
            return None  # Never qualify a search snippet, a URL-only source, or generated context as opened evidence.
        try:
            url = canonical_url(url)
        except ValueError:
            return None
        for evidence in self.evidence.values():
            if evidence.url == url:
                return evidence
        sid = f"S{len(self.evidence) + 1:03d}"
        item = Evidence(id=sid, url=url, title=str(source.get("title") or source.get("Title") or url),
                        text=raw[:24000], content_hash=hashlib.sha256(raw.encode()).hexdigest())
        self.evidence[sid] = item
        self.record("evidence_discovered", evidence_id=sid, url=url, content_hash=item.content_hash,
                    captured_characters=len(item.text), full_characters=len(raw))
        return item


class StateStore:
    def __init__(self, directory: str | Path | None = None):
        self.root = Path(directory or os.getenv("RESEARCH_STATE_DIR", "data/research"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "research.sqlite3"
        self.lock = None
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, updated REAL NOT NULL, payload TEXT NOT NULL)")

    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def open(self, query: str, settings: dict) -> ResearchState:
        fingerprint = hashlib.sha256(json.dumps(["adaptive-consumer-v1", query, settings], sort_keys=True).encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM runs WHERE fingerprint=? AND status IN ('running','interrupted') ORDER BY updated DESC LIMIT 1", (fingerprint,)).fetchone()
            state = ResearchState.model_validate_json(row[0]) if row else ResearchState(fingerprint=fingerprint, query=query)
            state._checkpoint_directory = str(self.root)
            self.lock = (self.root / f"{state.run_id}.lock").open("a")
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.lock.close()
                self.lock = None
                raise RuntimeError("This assignment is already running; keep its existing research page open.")
            if row:
                for task in state.investigations:
                    if task.status == "running":
                        task.status = "pending"
                state.record("resumed", completed_investigations=sum(t.status == "completed" for t in state.investigations))
            else:
                state.record("started", settings=settings)
            state.status = "running"
            self._save(db, state)
        return state

    def _save(self, db, state):
        ResearchState.model_validate(state.model_dump())
        db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)", (state.run_id, state.fingerprint, state.status, time.time(), state.model_dump_json()))

    def save(self, state: ResearchState) -> None:
        with self.connection() as db:
            self._save(db, state)

    def load(self, run_id: str) -> ResearchState:
        with self.connection() as db:
            row = db.execute("SELECT payload FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise KeyError(run_id)
        state = ResearchState.model_validate_json(row[0])
        state._checkpoint_directory = str(self.root)
        return state

    def close(self):
        if self.lock:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None
