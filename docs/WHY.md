# Why it's built this way

## The problem, in two sentences

AI agents are now given tools that send, book, update and spend, and the model
deciding when to call them can be wrong, confused or manipulated by text it reads.
Most agent code calls those tools directly, so there is nothing between "the model
asked" and "it happened": no allow-list, no human check, no limit, no reliable record.

The principle behind the library: **read freely, write carefully.** Actions that send
or write need stronger controls than reads.

## Design choices

**Why a library in the agent's process, not a proxy or a hosted gateway?**
Because the control has to sit where the tool function is called, with the
function's real arguments, and it has to work with any framework or none. A small
library with no server is also easier to read in full and to reason about, which
matters for something people are asked to trust.

**Why deny by default, and why can't `default_mode` be `allow`?**
Because the safe failure for an unknown tool is "no". A new tool added to the agent
without a policy entry should be refused, not silently allowed. Allowing unattended
execution is a decision to make per tool, in writing.

**Why four modes (allow, draft, approve, block)?**
Because the useful question is not "is this tool safe?" but "how much human
involvement does this action need?". Reading a calendar needs none. A reply can be
drafted and shown to the user. Sending to a team needs someone to look first.
Deleting a mailbox should never happen from an agent. Draft is its own mode because
"prepare but do not send" is often the right answer and is safer than approval: there
is nothing for a tired reviewer to wave through.

**Why return `Outcome` objects instead of raising exceptions?**
Because the agent loop has to tell the model what happened. "Queued for human
approval, id=act_..., do not retry" is a tool result the model can act on; a stack
trace is not. Exceptions are kept for programming and operator errors, and
`Outcome.raise_for_status()` is there for callers who prefer them.

**Why re-validate approved actions at execution time?**
Because approval and execution happen at different times. The human approved the
action in the world as it was; it should run only if it is still acceptable in the
world as it is. Between the two, the policy may have been tightened, the allow-list
changed, the budget spent, the approval gone stale or the kill switch pulled. So
everything is checked again, against the current policy, just before the tool runs.
This is the classic time-of-check to time-of-use problem.

**Why bind an approval to a digest of the arguments?**
Because a reviewer approves specific arguments, not a tool. If the stored arguments
no longer match the digest recorded when the action was queued, or if the current
policy's schema would change them (for example by adding a default), the action is
refused and must be requested again. What was approved is exactly what runs.

**Why SQLite?**
Because it is in the standard library, survives restarts, and gives real
transactions. The queue, the rate-limit counters, the budget and the duplicate check
all live in one table, so one `BEGIN IMMEDIATE` transaction can check every limit
and reserve the execution atomically. Two workers cannot both squeeze under the last
unit of budget. The concurrency tests show this with 24 parallel callers.

**Why do failed calls count against limits but not block retries?**
Because a failure is ambiguous: the email may or may not have gone before the
timeout. Counting it against the rate limit and budget is the cautious choice.
Blocking every retry, though, would turn a transient network error into a permanent
refusal, so failed actions are not treated as duplicates. This trade-off is written
down in the threat model rather than hidden.

**Why normalise arguments before de-duplicating, and why only lightly?**
Because retries from a model are rarely byte-identical: key order, trailing spaces,
Unicode composition and the order or case of recipients vary. Those are normalised.
Rewording is not, because deciding that two differently worded messages "mean the
same" is a judgement the library should not make silently. The key ignores which
agent asked, because the duplicate email is just as unwelcome either way.

**Why a hash chain for the audit log, not signatures or a database?**
Because it needs no keys and no service, and any change to a record, any deletion,
insertion or reordering, breaks the chain in a way `audit verify` reports with a line
number. Its limit is stated plainly: someone who can rewrite the whole file can
recompute every hash, and cutting off the newest records leaves a valid chain. Both
are caught by keeping the head hash somewhere else (`audit head`), which is cheap.
Tamper-evident, not tamper-proof.

**Why write the log as ASCII-only JSON lines?**
Because some characters (U+2028, U+0085) count as line breaks for some readers, which
would split one record in two for downstream tools. The property-based tests found
this. Hashes are computed over the parsed record, so the file's encoding does not
affect verification.

**Why redact with `[REDACTED]` rather than a hash of the value?**
Because hashes of short or guessable secrets (a PIN, a password) can be brute-forced,
which would turn the audit log into a leak. For the same reason only an exception's
type, never its message, goes to the model and the log.

**Why can the kill switch be engaged three ways?**
Because the person who needs to stop an agent may not have access to where it was
deployed. An environment variable suits deployment tooling, a flag file suits an
operator on the host (and the CLI), and an API call suits the application itself. It
is checked on every call, before every approved action, and again immediately before
the tool function runs.

**Why do approved actions stay approved while the kill switch is on?**
Because during an incident you want to review what was about to happen, not lose it.
Approvals can be withdrawn with `queue reject` before the switch is released, or left
to expire.

**Why doesn't the CLI execute approved actions?**
Because deciding and doing are separate responsibilities. The reviewer's CLI needs
no credentials for the mail server; the application that holds them executes
approved actions, and re-checks them first. Permission to approve is not the same as
permission to act.

**Why a short hand-written loop for Claude rather than the SDK's tool runner?**
Because the guard, not the SDK, must decide whether each function runs, and because
some turns must run no tools at all: a `refusal`, or a `max_tokens` stop part-way
through a tool call (whose input may be truncated). The loop is short enough to read
in full. It routes every `tool_use` block through the guard, returns
all results in one message, flags blocked and failed calls with `is_error`, and opts
into server-side refusal fallbacks by default.

**Why pydantic and YAML, with unknown keys rejected?**
Because a policy is security configuration: a typo such as `recipents:` must fail
loudly, not be ignored. YAML suits people who review policies; Python models suit
richer argument schemas. Both end up as the same validated object.

**Why synchronous only, for now?**
Because the state lives in SQLite and the checks are fast, and an async wrapper that
got the ordering subtly wrong would be worse than none. Async tool functions are
refused at registration with a clear error instead of being silently mishandled.

## Questions worth asking

**"Prompt injection is the real threat. Doesn't this just move the problem?"**
It moves it to a place where it can be bounded. Nobody currently knows how to stop
a model from being talked into *asking* for something harmful; OWASP's own entry on
prompt injection says so. What can be controlled, deterministically and in code, is
what happens next: which tools exist, where messages may go, how many, how much,
and which ones a person must see first. The guard does not judge intent; it enforces
limits that hold whatever the model was told. That is also why it is no substitute
for keeping untrusted content away from agents that hold powerful tools.

**"Won't people just rubber-stamp approvals?"**
Some will, which is why approval is not the only control and not the default for
everything. Recipient allow-lists, rate limits, budgets and de-duplication apply
before a human is asked, so reviewers never see requests that could not run anyway.
Draft mode removes the approval step entirely where a draft is enough. `max_pending`
stops the queue from growing into something nobody reads, and `queue show` puts the
full arguments in front of the reviewer. Whether approvals are meaningful is still an
organisational question: measure how often reviewers reject or edit, not just how
many approvals there are.

**"What stops someone with access to the server from bypassing all of this?"**
Nothing in this library, and it does not pretend otherwise. Someone who can write to
the database can mark actions approved; someone who can write to the log can rewrite
it. The design aims to make that *detectable* rather than impossible: approvals are
bound to argument digests, every decision records the policy fingerprint, and the
audit chain plus an external anchor shows any rewrite. Host security, credential
scoping and keeping raw credentials away from the agent remain your job.

**"Why not use a framework's built-in human-in-the-loop feature?"**
Use it if it covers what you need. Most such features pause a run to ask a question;
fewer re-check the approved action against current policy at execution time, bind the
approval to exact arguments, de-duplicate retries, enforce budgets atomically across
workers, and leave a verifiable record. This library is small enough to use alongside
a framework, or to read as a checklist of what to look for in one.

## What's next

- Async support (`Guard.acall`) with the same ordering guarantees.
- Signed approvals, so an approval can be verified independently of the database.
- A shipping hook for audit anchors (for example, to object storage with retention).
- Per-recipient rate limits and "first contact" rules (a new recipient needs approval
  even when the tool is otherwise allowed).
- Adapters for other tool-calling interfaces, each with its own fake-client tests.
- A short report generated from the audit log: decisions by tool and reason,
  approval latency, and how often reviewers reject.
