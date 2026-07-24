# Vesper — Architecture & Design Decisions

This document explains *why* Vesper is built the way it is. The code shows the
what; this is the reasoning. Every decision below was a fork in the road with a
cheaper alternative that was rejected for a specific reason.

The through-line: **Vesper is a single-user operator with judgment, not a
multi-tenant product.** Almost every choice falls out of taking that seriously.

---

## 1. The planner contains no routing logic

**Decision.** The Planner is a generic LLM tool-calling loop. It holds *zero*
knowledge of what any specific tool does. It sees a list of `ToolSpec` schemas,
asks the model which to call, hands the call to the Guardian, runs it if allowed,
feeds the result back, and repeats until the model stops. Adding "play music" or
"do deep research" or "post to Slack" adds **no branch** to the planner.

**Why.** The classic assistant is a switchboard: `if intent == "weather": …;
elif intent == "email": …`. That design rots. Every capability touches the
central router, intents collide, and the file that decides *everything* becomes
the file no one wants to touch. By making the planner a fixed loop over a
registry, capability growth is **additive and local** — a new tool is a new file
that registers itself on import, and the planner never learns it exists as
anything but another schema entry. Across PC0–PC4 (dev tools, music, research,
scripts, automation, Slack/Discord, the morning routine) the planner was **never
edited**. That is the test of the abstraction, and it held.

**What it costs.** You give up hand-tuned control flow. The model, not a
state machine, decides call order — so correctness leans on good tool
descriptions and the Guardian, not on the planner constraining what's reachable.
For a single trusted user that is the right trade; for an adversarial one it
would not be, which is exactly what the Guardian and the remote session policy
exist to bound.

---

## 2. Tools are a registry, not features

**Decision.** Capabilities are `ToolSpec` entries in a `ToolRegistry`, registered
by import side-effect. A tool is `{name, description, JSON-schema params, tier,
handler | (target_agent, action), category, slow, confirm_summary}`. Native tools
provide a `handler`; MCP tools are bridged in and get the *same* shape. The
registry renders the OpenAI-style function schema the model plans over.

**Why a registry and not methods/features.**
- **Uniformity.** A Gmail MCP tool, a native `run_shell`, and a bus-routed system
  action are indistinguishable to the planner and the Guardian. One code path
  governs all of them — tiering, confirmation, tracing, slow/queued execution.
- **The MCP bridge is free plumbing.** Adding an entire product (GitHub, Spotify,
  Slack, Discord) is a *config* entry — `command`, `args`, `tiers`, `expose`. The
  bridge connects, lists tools, and registers them with zero new code. `expose`
  is an allowlist so dangerous server tools (merge, force-push, delete-channel)
  are simply never surfaced. When Slack and Discord collided on identical tool
  names, the fix was one prefix-on-collision rule in the bridge — not a change
  anywhere else.
- **`slow=True` is a property, not a plumbing job.** The registry marks a tool
  slow; the planner submits it to the task queue and replies immediately; the
  queue emits the completion observation ("…is ready"). Deep research became
  background work by flipping a boolean.

**What it costs.** A registry is weakly typed at the boundary (JSON schemas, not
Python signatures) and indirection makes "what actually runs" a two-hop lookup.
Worth it: the alternative is the planner knowing every capability, which is
exactly decision #1's failure mode.

---

## 3. The Guardian and its three tiers

**Decision.** Every tool declares a tier: `safe` (runs immediately), `confirm`
(emits a `ConfirmationRequestedEvent`, waits for an explicit approval, times out
to denial), or `dangerous` (same gate, but treated as the top of the scale and
denylist-checked). The Guardian is the *only* thing that executes this policy;
the planner never decides whether something is allowed.

**Why tiers, not a boolean.** "Requires confirmation: yes/no" can't express the
real gradient. Reading your inbox should never interrupt you (`safe`). Drafting a
reply or creating a calendar event should (`confirm`). Running an arbitrary shell
command is categorically different from both — it warrants the full command shown
**verbatim**, a denylist that refuses `rm -rf /` / `sudo` / `curl | sh` / writes
outside `$HOME` **even after you approve**, and no "remember this approval" option
ever. Three tiers map to three genuinely different risk postures.

**Why it's a separate component on the bus.** Confirmation is a
request/response over the event bus, not an inline callback. That is what lets
*any* surface answer it — the CLI prints `[y/N]`, the HUD shows a card, the
Discord remote posts a prompt and waits for a reaction — without the Guardian
knowing which surface is listening. The audit log (`data/audit.jsonl`) records
every non-safe outcome with who approved it.

**The session policy (v3).** The remote interface needed a *lower* ceiling than
local surfaces: dangerous tools disabled entirely, no exceptions. Rather than
teach the planner about "remote", the Guardian reads a task-scoped
`SessionPolicy` from a `contextvar`. The remote bridge wraps each turn in
`session_policy(allow_dangerous=False)`; the policy propagates through
`handle_user_text → planner → Guardian.check` within that async task and can't
leak into a concurrent local turn. Dangerous-tier calls under that policy are
denied at the gate — never even offered for confirmation. The enforcement lives
in the one component whose job is enforcement, and the planner stayed untouched
(decision #1, again).

---

## 4. Sensors are local-only

**Decision.** FocusSensor, CalendarSensor, and InboxSensor observe only the local
machine (frontmost app via the macOS API, EventKit, the Gmail MCP tool). They
publish events onto the bus and store nothing off-device. There is no telemetry,
no cloud analytics, no "usage" endpoint.

**Why.** The sensors exist to give Vesper judgment about *your* day — that a
third context switch happened, that a meeting is five minutes out, that a
deep-work block is starting. That signal is intimate. The moment it leaves the
machine it becomes a liability (a breach surface, a privacy question, a thing a
recruiter or a user is right to distrust). Keeping sensing local means the
proactive engine can be as nosy as it needs to be to be *useful*, because the
data never travels. It also means the feature works with no account, no network,
and no third party — which is the whole point of a personal operator.

**What it costs.** No cross-device continuity, no server-side aggregation. For a
single-user, single-machine assistant that's not a loss; it's the design.

---

## 5. The proactive call-out doctrine is enforced in code

**Decision.** When a sensor rule trips, Vesper may raise it — but the constraint
"say it once, respectfully, then drop it" is mechanical, not a prompt request.
Rules are cooldown-gated (`_cooldown_elapsed`), observations are consumed after a
single turn, the morning routine is hard-limited to once per calendar day, and a
"not now" defers it with a second "not now" cancelling for the day.

**Why.** A proactive assistant that relies on the model to "please don't nag" will
nag — LLMs are eager, and eagerness compounds across turns. Restraint that
matters must be structural. The cooldowns and once-per-day gates make nagging
*impossible*, not merely discouraged. This is the single most important thing
separating "an assistant with a personality prompt" from "an operator you'd
actually leave running."

---

## 6. The event bus and the one-Brain model

**Decision.** A single in-process pub/sub bus connects everything; one `Brain`
owns all state (conversation context, memory, the planner). Every interaction
surface — CLI, HUD, voice, Discord remote — is a **stateless client** of that one
Brain, attaching over the localhost gateway. There is exactly one place that
reads and writes conversation context: `Brain.handle_user_text`.

**Why.** Surfaces multiply (terminal → HUD → voice → phone). If each carried its
own state, they would desync, and "who has the real conversation history" becomes
an unanswerable question. One Brain, many dumb clients, means a new surface is a
rendering problem, not a state problem — and it's why voice, the HUD, and the
remote interface could each be added without touching the planner or the context
model.

---

## 7. Memory is SQLite + a local vector store

**Decision.** Short-term context is in-process; long-term memory is SQLite plus a
local Chroma vector store. A reflection pass (nightly and at session end) distils
durable facts/preferences/patterns from the day's conversation into semantic
memory; relevant memories are retrieved by similarity on each turn.

**Why not a hosted vector DB.** For one user, a managed vector service is cost,
latency, a network dependency, and a privacy question in exchange for scale you
will never need. Local semantic memory retrieves in milliseconds, works offline,
and keeps preferences ("protect the gym slot", "no vocals while coding") on the
machine that learned them. The reflection-then-retrieve loop is what makes the
context-aware music and the day planner feel like the assistant *knows you*,
without a training step or a cloud account.

---

## 8. Local-first over a production stack

**Decision.** Vesper runs as local processes: a Python app, MCP subprocesses,
Ollama as an LLM fallback, a localhost-only FastAPI gateway that **refuses** to
bind to anything but `127.0.0.1`. No Kubernetes, no managed queue, no auth
service, no multi-tenant database.

**Why this is the *right* engineering, not a shortcut.** A production stack solves
problems a single-user agent doesn't have — horizontal scale, tenant isolation,
zero-downtime deploys, RBAC. Paying that complexity tax with no tenants to
isolate and no scale to reach would be cargo-culting. The genuinely hard problems
here are different and are the ones the architecture actually spends its
complexity on: **graceful degradation** (Groq → Ollama → a calm "I'm having
trouble thinking right now"; a crashed MCP server dropping out of the schema and
reconnecting with backoff), **a trustworthy permission model**, and **restraint**.
Choosing local-first is choosing to spend the complexity budget where the value
is. The gateway's hard localhost bind encodes the honest boundary: remote access
exists (the Discord bridge), but exposing the raw gateway to the network would
require real auth that a single-user tool shouldn't pretend to have — so it
refuses, loudly, rather than shipping a false sense of security.

---

## 9. Two Python runtimes, on purpose

**Decision.** The main app runs on Python 3.9 (pinned by pyobjc/langgraph/chromadb);
the MCP servers run the official `mcp` SDK on 3.10+ in their own virtualenv. The
bridge speaks raw JSON-RPC 2.0 over stdio, which has no version requirement of its
own.

**Why.** Rather than let one dependency's version floor block the whole project,
the process boundary *is* the compatibility layer. MCP servers are subprocesses
anyway; letting them have their own interpreter costs nothing and unblocks
everything. It's a small decision that reflects the larger philosophy: put the
seam where it's cheap.

---

## The shape of it

Sensors feed a proactive engine that has the discipline to stay quiet. A generic
planner reasons over a registry it doesn't understand in specifics. A guardian —
the only component that enforces — decides what may run and asks when it must.
Memory makes it personal; the event bus makes every surface a thin client of one
Brain; local-first keeps it honest. None of the individual pieces are exotic. The
argument of this project is that **the composition, and the restraint, are the
hard part** — and that's where the design spends its effort.
