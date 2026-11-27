# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- OpenAI and Azure OpenAI Chat Completions adapter (`agent_guardrails.adapters.openai`,
  optional `[openai]` extra, and `examples/openai_assistant.py` with offline demo
  mode): exports OpenAI function tool schemas from `Policy`, strictly parses tool-call
  JSON arguments (failing closed through `Guard` on malformed JSON, duplicate keys,
  `NaN`/`Infinity`, and non-object values), redacts structured tool outputs in
  `role: "tool"` messages, and refuses tool execution on `content_filter`, `refusal`, or
  `length`-truncated tool calls.
- `verify_log(anchor=(count, head))` and `audit verify --anchor RECORDS:HEAD`: check
  that a live log still starts with the records anchored earlier. `audit head` prints
  the anchor. (`expected_head` only matches a log that has not grown since.)

### Fixed

- Expired `pending` and `approved` actions no longer act as duplicates or consume
  `max_pending` quota before `pending()` / `run_approved()` sweeps them; `Guard.call()`
  expires stale actions inside the same `BEGIN IMMEDIATE` transaction, and
  `pending()` / `approve()` respect shortened `approval_ttl_seconds` in the active policy.
- `Guard.execute_approved()` reports `Status.BLOCKED` (`Reason.NOT_APPROVED` /
  `Reason.APPROVAL_EXPIRED`) rather than `Status.DUPLICATE` when a reviewer rejects or
  expires an approved action mid-flight.
- `Guard.call()` blocks non-mapping `args` with `Reason.INVALID_ARGUMENTS` instead of
  raising `TypeError`, and records `policy_hash` and redacted `args` even for malformed
  tool names.
- `Policy.from_yaml()` handles single-line YAML/JSON policies longer than 255 bytes
  without raising `OSError` (`ENAMETOOLONG`), and `ArgSpec` validates non-empty,
  type-compatible `choices`.
- `AuditLog.head()` and `verify_log()` acquire a shared `flock` (`LOCK_SH`) while
  reading, and fail closed with `AuditIntegrityError` / `VerificationResult(ok=False)`
  if a tampered record contains escaped lone surrogates (`UnicodeEncodeError`).
- `agent-guardrails queue list` redacts `DEFAULT_REDACT_FIELDS` in summary output and
  `--json` output while keeping full arguments in `queue show`.
- `scrub()` skips values shorter than four characters so trivial secrets such as `"a"`
  or `"1"` cannot corrupt ordinary words in audit detail strings.
- `adapters.claude.run_tool_loop` defaults `server_side_fallback=False` so standard
  `client.messages.create` clients (Anthropic, Bedrock, Vertex) work without opting into
  beta headers.
- `examples/claude_assistant.py` resolves sibling `email_calendar_assistant` cleanly
  regardless of working directory or `sys.path`.
- Several processes creating a new store at the same moment could fail with
  "database is locked": while one process switches a new file to WAL, SQLite can
  fail another's switch at once, without waiting on the busy timeout. The store now
  retries that step with a short backoff, up to its timeout.

### Fixed (security review)

- Recipient domains must be plain ASCII hostnames: NUL bytes, URL delimiters, empty
  labels and Unicode could pass a `*.` allow-list entry. Malformed allow-list entries
  are a `PolicyError`.
- Tools receive exactly the arguments that were checked, even with an `args_model`
  that excludes or re-serialises a field.
- YAML constraints on list types apply to each item; constraints that cannot apply
  to a type are refused instead of ignored.
- Lone surrogates in arguments or tool names are blocked and audited instead of
  crashing the guard.
- De-duplication treats `10` and `10.0`, and differently written but identical
  recipient lists, as the same action.
- Redacted values no longer reach the audit log through block messages, allowed
  calls no longer store secrets in `queue.db`, and new state files are owner-only.
- The audit verifier rejects duplicate keys and `NaN` instead of accepting or
  crashing.
- `AGENT_GUARDRAILS_KILL` engages for any value except an explicit "off".
- Approved actions are closed atomically; a shortened approval TTL applies to
  approvals already given; decisions record the fingerprint of the policy that made
  them; rejections record the state they actually overrode.
- Budgets are compared without floating-point drift.

## [0.1.0] - not yet released

### Added

- Declarative `Policy` (Python or YAML) with `allow`, `draft`, `approve` and `block`
  modes, argument schemas, recipient allow-lists, rate limits, per-agent budgets,
  approval time-to-live, de-duplication windows and queue-size limits.
- `Guard.call`, `Guard.wrap`, `@guard.tool()` and module-level `@guarded`, returning
  `Outcome` objects suitable as tool results.
- Durable SQLite approval queue with atomic check-and-reserve for limits.
- Execution-time re-validation of approved actions against the current policy,
  allow-list, budget, rate limit, kill switch, expiry and argument digest.
- Idempotency over normalised arguments.
- Hash-chained JSONL audit log with redaction, `verify_log` and head anchors.
- Kill switch via environment variable, flag file or API.
- `agent-guardrails` CLI: `queue list/show/approve/reject`, `audit verify/head`,
  `kill on/off/status`.
- Optional Claude tool-use adapter (`[anthropic]` extra).
- Offline email and calendar assistant example; threat model with an OWASP Top 10
  for LLM Applications 2025 mapping.
