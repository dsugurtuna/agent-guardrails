# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
