# Adaptive consumer research deployment

Fork: https://github.com/gregalthoff1977/gpt-researcher
Upstream: https://github.com/assafelovic/gpt-researcher
Upstream base: `3582a6841fe7dc801e2a34e34ee408641aea9bce`.

## Execution path

The existing Deep Research interface calls `GPTResearcher.conduct_research` →
`DeepResearchSkill.run` → `AdaptiveController.execute`. Each wave searches through
existing retrievers, passes selected URLs through the existing scraper and URL
security checks, qualifies opened excerpts, and evaluates all accumulated state.
Search snippets alone never become opened evidence. Previously opened URLs and
completed query results are reused.

The evaluation proposes INVESTIGATE, DEEPEN, CHALLENGE and REDIRECT actions with
question, evidence and hypothesis references. Selection ranks expected information
gain × importance / effort, rejects duplicates and invalid references, and records
rejected/deferred actions. STOP cannot establish completion while a critical gap
or unresolved contradiction remains. Effort limits can still stop a partial run.
Depth now bounds evidence waves rather than automatic follow-up recursion.

Consumer accounts, primary research, official disclosures, secondary reporting,
opinion and unknown material remain distinct. Extracted quotes must occur in the
opened text. Unknown population, independence, methods, geography and dates remain
unknown. Firsthand accounts cannot establish population prevalence. Verified is
reserved for exact bounded factual extracts from opened original material;
interpretations never become Verified. Broad and causal claims require suitable
methods, measurement context and independent original corroboration.

The writer renders findings, interpretations, hypotheses, contradictions, coverage
and qualified sources from the same structured state used for Knowledge JSON.
There is no second unconstrained narrative generation step or report-length quota.

## Durable state and recovery

Railway uses one replica, one Uvicorn worker and one persistent volume at `/data`.
SQLite WAL transactions save each completed investigation and evaluation under
`/data/research/research.sqlite3`. OS file locks prevent duplicate active requests
for the same assignment and settings. The entry point initializes mounted
permissions, then drops root before serving the app.

After a process interruption, submit the identical prompt with identical research
settings. The interrupted run resumes, skips completed investigations, restores
completed search results and re-evaluates an interrupted wave. In-flight work that
never checkpointed may be retried. Completed/effort-limited partial runs are final;
submitting them again starts a new run. Time and available cost already consumed
are retained on resume. Changing settings or wording starts a new assignment.
Multi-replica distributed recovery is outside this SQLite implementation.

`/data/outputs` holds Markdown, Knowledge JSON, execution trace and state snapshots.
The app's existing `outputs` path points there. Report history uses
`/data/reports.json`. Checkpoints and report files survive application redeploys;
this does not provide an off-platform backup or a retention policy.

## Configuration

| Railway variable | Value | Meaning |
| --- | --- | --- |
| RETRIEVER | tavily | Existing Tavily integration |
| DEEP_RESEARCH_BREADTH | 3 | Maximum investigations per wave |
| DEEP_RESEARCH_DEPTH | 2 | Default maximum evidence waves |
| DEEP_RESEARCH_CONCURRENCY | 4 | Maximum simultaneous investigations |
| DEEP_RESEARCH_MAX_WAVES | 0 | Use depth; positive value explicitly overrides wave count |
| DEEP_RESEARCH_MAX_SECONDS | 1200 | Cumulative research-time allowance |
| DEEP_RESEARCH_MAX_SEARCHES | 40 | Maximum dispatched retriever searches |
| DEEP_RESEARCH_MAX_COST_USD | 5 | Stop new model calls after known model-cost subtotal reaches this |
| DEEP_RESEARCH_MIN_GAIN | 0.15 | Minimum expected information gain for proposed investigations |
| DEEP_RESEARCH_CALL_TIMEOUT | 180 | Planner/evaluator/qualification call timeout in seconds |
| MAX_SCRAPER_WORKERS | 3 | Existing resource limit |
| MAX_SEARCH_RESULTS_PER_QUERY | 5 | Default URLs per query; existing UI setting remains usable |
| SMART_TOKEN_LIMIT / STRATEGIC_TOKEN_LIMIT | 32000 | Reasoning-model completion headroom |
| LLM_KWARGS | {"max_tokens":32000} | Existing deployment model headroom |
| RESEARCH_DATA_DIR | /data | Persistent data root |
| RESEARCH_STATE_DIR | /data/research | SQLite/checkpoint directory |
| REPORT_STORE_PATH | /data/reports.json | Existing report-history location |

OPENAI_API_KEY, TAVILY_API_KEY, RESEARCH_USERNAME and RESEARCH_PASSWORD are retained
in Railway. Password must contain at least 16 characters. All UI, API, asset,
download and WebSocket routes require authentication; only GET/HEAD `/__health`
are public. Health includes `x-research-engine: adaptive-consumer-v1`.
Cross-origin mutations and WebSockets are rejected. Use HTTPS.

Build: `Dockerfile.railway`, Python 3.12, original browser/scraping dependencies.
Start: `python -m backend.server.production`.
Pre-deploy: `python scripts/verify_adaptive_deployment.py`.
The previous environment-supplied authentication bootstrap is no longer executed.
Source is pinned to the verified fork commit, so later branch pushes require an
explicit source update/redeployment. No additional research framework or service
was introduced. A volume attachment may briefly interrupt the single replica.

The cost allowance is a guard on **available model-cost data**, not a billing cap:
search fees, unavailable prices and calls already in flight can add cost. Trace
cost completeness is explicit. Token counts and unknown costs are never invented.

## Exports and schema

The JSON download is the authoritative Consumer Research Director v1.1 output
contract (its envelope requires `schema_version: "1.2"`). The actual supplied
contract and sanitized Notion property snapshot are in `gpt_researcher/schemas`.
Properties are Name, Type, Content, Source URL, Evidence, Topics, Entities,
Relationships, Confidence, Geography, Time Period, Implications and Status.

JSON Schema and reference validation run before publishing. Source IDs, claim
audits, confidence rationale, proposed vocabulary options and local relationships
are preserved. `Relationships` stays empty because existing Notion page UUIDs
were not verified; local K-record links appear in `proposed_relationships`.
Dates without an established exact start/end remain null. No Notion writes occur.

The UI adds Trace JSON and Research State download links. Trace events include
actual model messages/responses, search results, source qualification, findings,
hypothesis changes, candidate selection/rejection, stopping reasons and available
runtime/cost data. Environment secrets and recognizable API key patterns are
redacted from JSON exports; research content itself may contain sensitive material.
The trace records observable execution, not a model's private reasoning.

Separate latest-evidence completeness is not automatically certified.
`freshness_checks` says Not checked, retains incomplete freshness requirement IDs,
and completion review remains Partial until reviewed. This is explicit even when
substantive research questions have sufficient bounded evidence. Automated source
classification and confidence gates do not replace a human factual audit or prove
research quality.

## Automated verification

Python 3.12 offline suite: **607 passed, 1 skipped, 0 failed** across 136 modules.
The skip is the existing `test_security_fix.py` module whose helpers were removed
upstream. The same three live modules excluded by upstream CI remain excluded:
`test_researcher_logging.py`, `test_logging_output.py`, and `test_mcp.py`.
No billable/live research was used for development verification.

The 35 new tests cover adaptive selection using accumulated evidence, actual
Deep Research integration, firsthand/primary/secondary/opinion classification,
contradictions and competing hypotheses, unsupported-claim gates, recorded
rationales, SQLite restore without completed-search duplication, failed retrieval,
malformed evaluations, available-cost guard, redaction, exact Markdown/Knowledge
exports, final download paths, authentication, existing UI/assets and WebSocket
ping, mounted output paths and browser disconnect handling.

A 72-test focused regression set passes and also runs against the exact container
sources during image build. Pre-deploy readiness checks authenticate the actual
interface, load static assets, round-trip SQLite state and validate the exact
Knowledge schema. An outbound socket guard prevents readiness from performing
live research. `node --check frontend/scripts.js` passes.

Reproduce locally after installing requirements and pytest/pytest-asyncio:

```sh
python scripts/run_offline_tests.py
node --check frontend/scripts.js
RESEARCH_PASSWORD=offline-readiness-fixture-password python scripts/verify_adaptive_deployment.py
```

The runner isolates each module because upstream test fixtures replace global
modules; a single-process suite leaks those mocks. Updated fixtures use real
requests imports and mocked public DNS for canned scraper responses, without
changing production URL protections. GitHub Actions uses the isolated runner and
uploads per-module logs. Only Python 3.12 was executed locally; the existing CI
matrix also declares Python 3.13 and 3.14, whose remote results must be inspected
separately.

## User acceptance

Use the existing research URL and password, select Deep Research, and submit the
existing Chameleon Cold-Brew assignment. Keep Tavily, breadth 3, depth 2,
concurrency 4 and five websites per query for the baseline comparison. Preserve the
assignment wording used in prior runs. Inspect the report, Knowledge JSON and
Trace JSON together. Live research quality and model/provider access are left for
that independent acceptance test. Development stops after deployment verification.
