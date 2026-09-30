# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `verify_log(anchor=(count, head))` and `audit verify --anchor RECORDS:HEAD`: check
  that a live log still starts with the records anchored earlier. `audit head` prints
  the anchor. (`expected_head` only matches a log that has not grown since.)

### Fixed

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
