# Policy reference

A policy is validated strictly: unknown keys, wrong types and inconsistent settings
raise `PolicyError` when it is loaded. See [`examples/policy.yaml`](../examples/policy.yaml)
for a complete, commented example.

```python
from agent_guardrails import Policy

policy = Policy.from_yaml("policy.yaml")  # a path, or a YAML string
policy = Policy.from_dict({...})  # a plain dict
policy.fingerprint()  # SHA-256, recorded with every decision
```

## Top level

| Key | Default | Meaning |
|---|---|---|
| `version` | `1` | Schema version. |
| `default_mode` | `block` | Mode for tools not listed under `tools`. `allow` is refused: unattended execution must be granted per tool. |
| `tools` | `{}` | Per-tool policy, keyed by tool name. |
| `budgets` | `{}` | Spend limits per agent id. The key `"*"` applies to agents not listed. Agents with no budget are unlimited. |
| `redact_fields` | common secret names | Argument keys (case-insensitive, at any depth) replaced by `[REDACTED]` in the audit log; their values are also removed from the detail text of blocked-call records. Keys match exactly: `secret` does not cover `client_secret`, so list each name. |

### `budgets.<agent>`

| Key | Meaning |
|---|---|
| `limit` | Maximum total cost (any unit: pounds, credits, messages), compared to nine decimal places so that floating-point error does not eat into it. |
| `window_seconds` | Rolling window. Omit for a lifetime limit (per state directory). |

## `tools.<name>`

| Key | Default | Meaning |
|---|---|---|
| `mode` | required | `allow`, `draft`, `approve` or `block`. |
| `description` | none | Free text for humans. |
| `args` | none | YAML argument schema (below). Unknown arguments are rejected. |
| `args_model` | none | Python only: a pydantic model class, instead of `args`. |
| `recipients` | none | Recipient allow-list (below). |
| `rate_limit` | none | `{max_calls, window_seconds, per_agent}`. Counts calls that ran (including failed ones) in the trailing window, across all agents unless `per_agent: true`. |
| `cost` | none | `{fixed: 0.5}` or `{field: amount}`. Charged to the calling agent's budget. A `field` value must be a finite number >= 0. |
| `approval_ttl_seconds` | `3600` | How long a queued action stays valid, from when it was requested. After this it can be neither approved nor run. |
| `dedupe_window_seconds` | `3600` | Identical actions within this window are not repeated. `0` disables de-duplication. While on, an identical action that is still pending or approved counts as a duplicate however old it is. |
| `max_pending` | none | Maximum number of this tool's actions awaiting approval at once. |
| `redact_fields` | `[]` | Extra argument keys to redact for this tool (for example `body`). |

### `args.<argument>`

| Key | Applies to | Meaning |
|---|---|---|
| `type` | all | `str`, `int`, `float`, `bool`, `list[str]`, `list[int]`, `list[float]`. Default `str`. |
| `required` | all | Default `true`. |
| `default` | all | Value used when not required and not given. |
| `choices` | all | Allowed values (for lists, allowed values of each item). |
| `min_length`, `max_length`, `pattern` | `str`, `list[str]` | String constraints; for a list, each item must meet them. `pattern` is a regular expression that is *searched for*, not matched against the whole value: write `^...$` to anchor it. |
| `ge`, `le` | `int`, `float`, `list[int]`, `list[float]` | Numeric bounds; for a list, on each item. |
| `min_items`, `max_items` | lists | List length bounds. |

A constraint that cannot apply to the type (for example `pattern` on an `int`, or
`ge` on a `list[str]`) is a `PolicyError`, not silently ignored.

For anything richer (nested objects, dates, custom validators) use `args_model` with a
pydantic model in Python. Set `model_config = ConfigDict(extra="forbid")` on it if
unknown arguments should be rejected.

### `recipients`

| Key | Meaning |
|---|---|
| `fields` | Argument names holding addresses (a string, a comma-separated string, or a list). All are counted and checked together. |
| `allowed_domains` | `example.com` matches exactly; `*.example.com` matches subdomains only; `*` matches everything (write it on purpose). Entries must be ASCII hostnames; write internationalised domains in their `xn--` form. A malformed entry is a `PolicyError`. |
| `max_recipients` | Maximum total addresses across `fields`. |

Addresses are parsed strictly: each value must yield exactly one address per `@`,
so a display name hiding a second address (`"a@evil.test" <b@example.com>`) is
rejected rather than guessed at. Values containing control or invisible characters
(NUL, line breaks, zero-width spaces) are rejected, and the domain must be a plain
hostname: letters, digits and hyphens in non-empty labels, with at most one trailing
dot. So `a@evil.test#.example.com` or `a@evil.test\x00.example.com` cannot pass a
`*.example.com` entry by being read differently later. Unicode domains are refused,
because `str.lower()` and IDNA do not map them the same way; use the `xn--` form.

## Order of checks

For a new call: tool name (a non-empty string of valid Unicode), kill switch, mode,
arguments (bound to the function signature, then validated), recipients, cost; then per mode: `draft` returns a preview; `approve`
checks duplicates, `max_pending` and whether the cost exceeds the whole budget, then
queues; `allow` checks duplicates, rate limit and budget and reserves the execution in
one transaction, re-checks the kill switch, and runs.

For an approved action (`Guard.execute_approved` / `Guard.run_approved`): status,
kill switch, expiry, argument digest, current mode, registered function, arguments
and recipients against the current policy, whether the current policy would change
the arguments, then duplicates, rate limit and budget in one transaction, then a
last kill-switch check, then the tool runs.
