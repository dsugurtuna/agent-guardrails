# Threat model

> Unofficial. This is a personal open-source project, not government guidance and not
> OWASP guidance. The OWASP mapping below is the author's reading of public OWASP
> material; it is not an endorsement or a certification.

## The system in one paragraph

An AI agent (a model in a tool-use loop) can call tools. Some tools only read
(list calendar events). Others have side effects (send an email, create an event,
spend credits). agent-guardrails sits between the agent loop and the tool functions:
every call goes through `Guard.call`, which applies a declarative policy, keeps
durable state in SQLite and writes a hash-chained audit log. People supervise it
through a CLI: they approve or reject queued actions, verify the log and pull the
kill switch.

## Trust boundaries and assumptions

| Component | Trusted? | Why |
|---|---|---|
| Model output (tool names, arguments) | **No** | It can be wrong, confused, or steered by injected text in emails, web pages or documents it reads. Treated like input from an anonymous user. |
| The policy file | Yes | Written and reviewed by the people who own the deployment. Its SHA-256 fingerprint is recorded with every decision. |
| The guard process and tool functions | Yes | Your code. The guard cannot protect a tool the agent can reach by another path. |
| The reviewer | Yes, but fallible | People approve things they should not, especially when tired or rushed. |
| Queue database, audit log, kill-switch flag | Protected by OS permissions | The library cannot defend them against someone who already has write access to the host (see "What it does not defend against"). |

The key assumption is **complete mediation**: the only way for the agent to cause a
side effect is through a guarded tool. If the agent also holds raw API credentials,
a shell, or a general-purpose code-execution tool, the guard can be bypassed.

## What it defends against

| Threat | Example | Control | Tested in |
|---|---|---|---|
| The agent requests a harmful write | An injected instruction in an email makes the agent send the user's calendar to an outside address | Deny-by-default tool list; per-tool modes; recipient domain allow-list with strict address parsing | `test_modes.py`, `test_checks.py` |
| Too much autonomy for high-impact actions | The agent sends a message to a whole team without anyone reading it | `approve` mode: queued for a human; nothing runs until approved | `test_approvals.py` |
| Drafts mistaken for sends | The model "sends" a reply that should only have been drafted | `draft` mode never calls the function; the tool result says so in plain words | `test_modes.py` |
| Unbounded consumption | A loop sends 500 SMS messages or books 200 meetings | Rate limits per tool (optionally per agent); spend budgets per agent; limit on pending approvals | `test_limits.py`, `test_concurrency.py` |
| Duplicate side effects | The agent retries after a timeout and sends the same reminder twice | Idempotency key over normalised arguments; identical actions within the window are not repeated, including while one is awaiting approval | `test_idempotency.py`, `test_concurrency.py` |
| Stale approval (time-of-check vs time-of-use) | An email approved on Monday runs on Wednesday, after the recipient's domain was removed from the allow-list and the budget was spent | Approval TTL; full re-validation against the *current* policy, allow-list, budget, rate limit and kill switch before running | `test_approvals.py` |
| Approved arguments changed before execution | A bug or a person edits the queued row after approval | The approval is bound to a SHA-256 digest of the arguments; a mismatch blocks execution. A policy change that would alter the arguments also blocks it | `test_approvals.py` |
| Argument smuggling | `"a@evil.test" <b@example.com>`; `a@evil.test\x00.example.org` or `a@evil.test#.example.org` against a `*.example.org` entry; an extra `bcc` field; a negative payment amount to "refund" the budget | Addresses must parse unambiguously (one address per `@`), contain no control or invisible characters, and have a plain ASCII hostname as the domain; schemas forbid unknown fields; cost fields must be finite and non-negative | `test_checks.py`, `test_limits.py` |
| Races between parallel callers | Two workers both squeeze under the last unit of budget | Check-and-reserve inside one `BEGIN IMMEDIATE` SQLite transaction; compare-and-set when claiming an approved action | `test_concurrency.py` |
| Silent edits to the record | Someone deletes the entry showing an action ran, or reorders entries | Hash-chained JSONL audit log; `audit verify` detects edits, deletions, insertions, reordering and ambiguous records (duplicate keys); an anchor kept elsewhere (`audit head`, later `audit verify --anchor`) also detects removal of the newest records and whole-file rewrites | `test_audit.py`, `test_audit_properties.py` |
| Secrets in logs and tool results | An API key in tool arguments; a password inside an exception message | Redaction of named fields (defaults plus per-tool) and a redaction hook; only the exception *type* goes back to the model and into the log | `test_modes.py` |
| An agent misbehaving right now | Reports of odd messages going out | Kill switch (environment variable, flag file or API), checked on every call, before every approved action and immediately before each tool function runs. The variable fails closed: any value that is not clearly "off" engages it | `test_killswitch.py` |

## What it does not defend against

Being clear about this matters more than the list above.

- **Prompt injection itself.** The model can still be manipulated into *requesting*
  bad actions, and into telling the user misleading things. The guard limits what
  those requests can do; it does not detect or prevent the injection.
- **Harmful content inside a permitted action.** An email to an allow-listed
  colleague that contains wrong or manipulated content passes every check. Content
  review is the reviewer's job in `approve` mode.
- **Exfiltration through permitted channels.** If an allow-listed domain is
  compromised, or a read tool puts sensitive data in the model's context and an
  allowed write tool carries it out, the allow-list narrows but does not close
  that path.
- **Reads.** Read tools are not controlled unless you wrap them, by design
  ("read freely"). Reading can still expose sensitive data to the model.
- **Bypass paths.** Anything the agent can reach without going through the guard.
- **Secrets in the queue database.** An action queued for approval keeps its full
  arguments in `queue.db`, including fields redacted from the audit log, because it
  must later run with its real arguments; `queue show` displays them to reviewers.
  Actions run at once keep only redacted arguments. Nothing is purged yet: protect
  `queue.db` like a credential store, and prefer tools that look secrets up
  themselves over tools that take them as arguments.
- **A compromised host.** Anyone with write access to the SQLite file can mark
  actions approved. Anyone with write access to the audit log can rewrite it and
  recompute every hash; only an anchor kept elsewhere (`audit head`, then later
  `audit verify --anchor RECORDS:HEAD`) detects that, and truncation of the newest
  records. An anchor covers only the records written before it was taken: records
  appended since are protected by the chain alone, and someone who can write to the
  log can append well-formed forged records after the anchor. Anchor regularly, to
  storage the application cannot write. Anyone who can edit the policy file can
  change the policy (the fingerprint in each record shows when it changed).
- **Reviewer error and approval fatigue.** Approvals can be rubber-stamped. Keep
  `approve` mode for actions that deserve a human, cap the queue with `max_pending`,
  and review full arguments with `queue show`.
- **Reviewer identity.** The CLI records `--by` as given (default: the OS user). It
  does not authenticate anyone. Put approval behind your own identity system if the
  approver's identity matters.
- **Paraphrased duplicates.** De-duplication catches identical actions, with
  whitespace, Unicode form, key order, number form (`10` and `10.0`) and recipients
  (case, order, repeats, display names, list or comma-separated string) normalised.
  It does not catch the same message reworded.
- **Ambiguous failures.** If a tool times out after the side effect happened, a
  retry may repeat it: failed actions are not de-duplicated (so genuine failures can
  be retried), although they do count towards rate limits and budgets. If the process
  dies mid-call, the action stays `executing`; identical retries are refused within
  the de-duplication window and allowed after it.
- **Clock manipulation.** TTLs and windows use the system clock.
- **The kill switch is cooperative.** It stops actions that pass through a guard
  that checks it. It does not stop processes or revoke credentials; pair it with
  credential revocation in a real incident.
- **Windows multi-process writers.** The audit log uses POSIX `flock` across
  processes. On Windows only threads within one process are serialised.

## Why re-validate at execution time?

Approval and execution happen at different times, often in different processes.
Checking once, at request time, leaves a time-of-check to time-of-use gap: the
human approved the action in the world as it was, but it runs in the world as it
is. In between, the policy may have been tightened, the allow-list changed, the
budget spent by other actions, the approval may have gone stale, or someone may
have pulled the kill switch. So `Guard.execute_approved` repeats every check against
current conditions, confirms the stored arguments still match the approved digest,
and refuses to run if the current policy would change those arguments. Transient
reasons (kill switch, rate limit, budget) leave the action approved so it can run
later or be withdrawn; permanent ones close it.

## Mapping to the OWASP Top 10 for LLM Applications 2025

**How this was verified.** The owasp.org and genai.owasp.org sites were not
reachable from the environment this was written in, so the entries were read from
the official OWASP source repository instead:
[GenAI-Security-Project/GenAI-LLM-Top10](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10),
directory `2025/`, at commit
[`9253e38`](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/tree/9253e38ade58e959b531c0c5c9a4842272c9cd0e).
The canonical publication page is <https://genai.owasp.org/llm-top-10/>.
That repository states that a **2026 edition was published on 4 August 2026** and
reorders the list. Where an entry keeps the same name in the 2026 edition, its 2026
ID (read from `2026/final/` at the same commit) is shown alongside. The mitigation
numbers below refer to the 2025 entries.

| 2025 entry | 2026 ID | What the entry recommends (paraphrased) | Controls here | Not covered here |
|---|---|---|---|---|
| [LLM06:2025 Excessive Agency](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2025/LLM06_ExcessiveAgency.md) | LLM03:2026 | 1 minimise extensions; 2 minimise extension functionality; 6 require user approval for high-impact actions; 7 complete mediation (authorisation in downstream systems, not in the LLM); log and monitor; rate-limit | Deny-by-default tool list (1); per-tool modes, e.g. read `allow`, send `approve` (2); `approve` mode (6); the guard as a policy decision point in code, independent of the model (7); audit log; rate limits | 3 avoid open-ended extensions and 4 minimise extension permissions are tool and credential design; 5 execute in the user's context is the tool's job |
| [LLM10:2025 Unbounded Consumption](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2025/LLM10_UnboundedConsumption.md) | LLM06:2026 | 3 rate limiting and quotas; 4 resource allocation management; 7 logging and monitoring; 10 limit queued actions and total actions | Rate limits per tool, optionally per agent (3); spend budgets per agent (4); audit log (7); `max_pending` (10) | Token and compute limits on the model itself, model extraction, watermarking |
| [LLM01:2025 Prompt Injection](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2025/LLM01_PromptInjection.md) | LLM01:2026 | The entry itself says it is unclear whether fool-proof prevention exists; 4 enforce privilege control and least privilege, handling functions in code; 5 require human approval for high-risk actions | Limits the *impact* of injected requests: policy enforced in code (4); `approve` mode (5) | Detection and prevention of injection; input and output filtering; segregating untrusted content |
| [LLM02:2025 Sensitive Information Disclosure](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2025/LLM02_SensitiveInformationDisclosure.md) | LLM02:2026 | Sanitisation; access controls; following API security misconfiguration guidance to avoid leaking information through error messages | Redaction before logging; exception messages withheld from the model and the log; recipient allow-lists narrow where data can be sent | Training-data sanitisation, access control to data sources, privacy techniques |
| [LLM05:2025 Improper Output Handling](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2025/LLM05_ImproperOutputHandling.md) | LLM10:2026 | 1 treat the model as any other user and validate its responses before they reach backend functions | Strict argument schemas (unknown fields rejected), unambiguous address parsing, numeric checks on cost fields | Output encoding for HTML, SQL or shells (the tool's job) |

Not addressed by this library: LLM03:2025 Supply Chain, LLM04:2025 Data and Model
Poisoning, LLM07:2025 System Prompt Leakage, LLM08:2025 Vector and Embedding
Weaknesses and LLM09:2025 Misinformation.

The 2026 Excessive Agency entry (LLM03:2026) adds detail that matches this design
closely: an "independent pre-execution policy decision point" for complete
mediation, graduated enforcement that routes irreversible actions to human review,
and thresholds or circuit breakers based on the number of invocations or "the
cumulative value of an input parameter" (which is what a budget with a `cost.field`
is). See
[`2026/final/LLM03_ExcessiveAgency.md`](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10/blob/9253e38ade58e959b531c0c5c9a4842272c9cd0e/2026/final/LLM03_ExcessiveAgency.md).
