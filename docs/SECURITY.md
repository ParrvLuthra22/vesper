# Security

The threat model for Vesper: what it protects, who and what it does not trust, the defense for each threat, **where that
defense is tested**, and what is left over. It is written from the code and the test suite, not from intent — a defense
with no test is listed as untested.

Reporting a problem: open an issue **without** pasting real mail, messages or keys (use `vesper briefing --redact`; send
counts and file names). See [LIVE_VERIFICATION.md](LIVE_VERIFICATION.md) for the evidence rules.

## What is being protected

| Asset | Where it lives |
|---|---|
| The ability to act on your behalf (open/close apps, run commands, write drafts, post) | the tool registry, gated by the Guardian |
| Your mail, calendar, Slack and Messages content | Google / Slack / `chat.db` (read-only access), the local cache `data/briefing.db` |
| Credentials: Groq key, Gmail OAuth token, Slack user token, Telegram bot token | `.env`, `data/google_token.json`, macOS Keychain |
| The audit trail | `data/audit.jsonl` (append-only, local) |

## Who is trusted how

| Source | Trust | Why |
|---|---|---|
| You, typing or speaking at the Mac (CLI, voice, HUD) | trusted | the only surfaces that can approve outbound actions |
| You on Telegram / Discord | **restricted** | an account takeover or a stolen unlocked phone is "you" to the bot; so dangerous tools are disabled and outbound approvals need the Mac |
| Text inside email, Slack, Messages, calendar entries, web pages, forwarded chat messages | **untrusted data** | written by someone else; may contain instructions aimed at the model |
| The planner (the LLM) | not trusted to restrain itself | permissions are enforced outside it |
| MCP servers and the local machine | trusted (out of scope) | malware on the Mac, or a malicious MCP server you enabled, is not defended against |

## Threats, defenses, tests

### 1. Prompt injection through content (email, Slack, Messages, calendar, web, forwarded text)

| Defense | Where | Tested in |
|---|---|---|
| Hard truncation; URLs, addresses, markup and shell substitutions neutralised at collection | `briefing/sanitize.py`, `briefing/slack.py`, `briefing/messages.py` | `test_briefing_builder.py`, `test_slack_collector.py`, `test_messages_collector.py` |
| Text that reads like instructions to an AI stores **no snippet**, is demoted, never named, and is reported only as a count ("N … filtered as unsafe") for mail, Slack and Messages | scorer + `briefing/builder.py` | `test_gmail_flagged_line.py`, `test_slack_flagged_line.py`, `test_messages_collector.py` (hostile-only and mixed inboxes) |
| Third-party strings reach the model only as quoted JSON string literals in a block that says they are data | `briefing/builder.py` | `test_slack_briefing.py` (quoting, `$(…)` payloads), `test_briefing_builder.py` |
| The tool that carries the block is `untrusted_output`, which **taints** the turn | `orchestrator/brain.py`, `tools/registry.py` | `test_messages_collector.py::test_the_briefing_tool_stays_untrusted_output`, `test_guardian_taint.py` |
| **Tainted-input rule**: a later call with an argument the user did not say is raised one rung (`safe→confirm`, `draft→confirm`, `confirm→dangerous`) | `guardian/gate.py`, `orchestrator/planner.py` | `test_guardian_taint.py`, `test_planner.py`, `test_injection_regression.py`, `test_channel_gateway.py` (forwarded text) |
| A recipient the user never said raises the tier even if nothing untrusted was read this turn | `ToolSpec.recipient_args` | `test_gmail_draft.py` |
| Forwarded / third-party chat text never takes the router shortcut | `orchestrator/brain.py` | `test_router.py::test_unmatched_and_third_party_turns_still_reach_the_planner` |
| Tools never interpolate model-supplied values into a shell or AppleScript string (argv only; URLs limited to http/https) | `utils/safe_exec.py` | `test_injection_regression.py` (27 tests, the 2026-10-02 audit's injection class) |

**Left over:** none of this makes injection impossible. The rule is turn-level and one rung — taint does not follow replayed
conversation history into a later turn ([ARCHITECTURE.md](ARCHITECTURE.md) §4). A persuaded model can still *ask* for a
confirm-tier action; the defense is that it must show you what it will do and wait.

### 2. A compromised remote channel (Telegram, Discord) — or a stolen unlocked phone

| Defense | Where | Tested in |
|---|---|---|
| Allowlist by numeric id, enforced in the adapter and again at the gateway; strangers get no reply | `channels/telegram.py`, `gateway/channels.py` | `test_telegram_channel.py`, `test_channel_gateway.py` |
| Remote sessions are always restricted: `dangerous` tools are refused outright | `guardian/gate.py` (session policy) | `test_channel_gateway.py`, `test_reach.py` |
| Approve buttons are bound to a single-use, user-bound, expiring nonce; replay, stale and foreign-user presses are refused and audited | `channels/confirm.py` | `test_channels_core.py`, `test_telegram_channel.py` |
| **Outbound-action rule**: anything that delivers content to another person, or writes a draft, and was raised remotely can only be **approved on the Mac** (HUD / CLI); the remote card has no Approve button, the door answers `needs_mac`, the Guardian refuses a non-local approver and audits `channel:approval_needs_mac`; remote **deny** always works | `guardian/gate.py`, `gateway/channels.py`, `channels/telegram.py`, `remote/discord_remote.py` | `test_outbound_approval.py` (25 tests: refusal per channel, local approval, replays, foreign users, expiry, the HTTP door) |
| MCP tools that post to others (`post_message`, `reply_thread`, `add_issue_comment`) are flagged `outbound`; an outbound tool can never be `safe` | `tools/registry.py`, `tools/mcp_bridge.py`, `config/settings.yaml` | `test_mcp_bridge.py`, `test_outbound_approval.py` |
| Every non-safe verdict is audited with the channel and the approver | `guardian/gate.py` | `test_guardian.py`, `test_outbound_approval.py` |

**Left over:** an attacker with your Telegram account can do everything *a restricted session* allows, including approving
local-only confirm-tier actions. Use Telegram's two-step verification and a passcode. The approver's `source` string is
set by the in-process component that emits the event (the gateway stamps `"gateway"` only on its authenticated local
endpoint); the **process** is the trust boundary, there is no cryptographic signature. There is no voice-approval path
today (the HUD card and the CLI are the local approvers).

### 3. Reading private data without exfiltrating or mutating it

| Defense | Where | Tested in |
|---|---|---|
| Collectors reach tools only through an allow-list of pure reads (`safe` tier) | `briefing/readonly.py` | `test_briefing_collectors.py` |
| Slack: a six-method allow-list; every mutating method is refused **before a request is built**; the token travels only as a header and is scrubbed from errors | `briefing/slack_api.py` | `test_slack_readonly.py` (46 mutating methods, smuggled names, a source scan) |
| Messages: `chat.db` is **copied** to a private `0700` temp dir, opened `mode=ro` + `PRAGMA query_only`, and deleted afterwards (also on failure); each layer is asserted on its own; the original's hash and mtime are unchanged; off by default and not touched unless enabled | `briefing/messages.py` | `test_messages_collector.py` (writes must raise, WAL visible, no temp left behind) |
| One-time codes (any 4+ digit run) are masked **before** truncation, so a code never reaches a snippet, the cache, the planner block or speech | `briefing/messages.py` | `test_messages_collector.py`, `test_messages_redact.py` |
| `vesper doctor` checks Full Disk Access by **opening** the file only — no byte is read | `launcher/doctor.py` | `test_doctor.py` (asserts `os.read` is never called) |
| `gmail_create_draft` can only create a draft: the backend interface has one method, a test fails if the module's source mentions delivering mail, and the Gmail client/server are scanned for any delivery call; caps and header-injection checks run first | `tools/gmail_draft.py`, `mcp_servers/gmail/` | `test_gmail_draft.py` (39 tests, mocked Gmail only) |

**Left over:** the Messages collector and the draft tool have only ever run against synthetic databases and mocks — no real
`chat.db` was opened and no real draft was created in the course of building them.

### 4. Secret and data leakage

| Defense | Where | Tested in |
|---|---|---|
| `--redact` fails closed: the CLI formats only placeholders, and the whole output is scanned against everything sensitive stored locally and withheld on a hit; only whole lines matching a declared static shape are exempt from the scan | `briefing/redact.py`, `briefing/cli.py` | `test_briefing_redact.py`, `test_slack_redact.py`, `test_messages_redact.py`, `test_redact_static_text.py` (canaries, simulated layout change) |
| Notifications and proactive interruptions can carry only a source label, a sanitised sender display name, or minutes-to-meeting — never a subject, snippet, body or meeting title | `proactive/watcher.py` | `test_proactive_watcher.py` (a fixed whitelist of shapes against canary content) |
| Secrets stay out of git: `data/`, `.env*`, tokens and the briefing cache are git-ignored. The Telegram bot token is kept in the Keychain/env and scrubbed from every log handler; the Slack token is scrubbed from error text and only ever sent as a header | `.gitignore`, `channels/credentials.py`, `briefing/slack_api.py` | `test_gitignore_rules.py`, `test_channels_core.py`, `test_telegram_channel.py`, `test_slack_readonly.py` |
| What leaves the machine is enumerated | [PRIVACY.md](PRIVACY.md) | read from code; **not packet-captured** |

**Left over:** the planner prompt (including the tool results in it) goes to Groq; Groq's key is a plain environment value
(rotate it if it was ever pasted anywhere). Redaction protects *printed output*, not the cache file itself — treat
`data/briefing.db` as sensitive.

### 5. Autonomy gone wrong (an assistant that does too much, or too loudly)

| Defense | Where | Tested in |
|---|---|---|
| Deterministic router: whole-phrase patterns, one skill per utterance, validated arguments, never closes its own host or sweeps, every call through the Guardian at `max(skill, tool)` tier | `router/` | `test_router.py` and `tests/eval_router.py` (100 % precision on 59 utterances that must fall through) |
| Proactive watcher: hard daily budget, quiet hours, cooldowns, persisted dedupe, DND; off by default | `proactive/watcher.py` | `test_proactive_watcher.py` |
| An action claim ("I've archived it") is checked against the tools that actually succeeded | `orchestrator/claim_verifier.py` | `test_claim_verifier.py` |

## How these checks were validated

For the Messages snapshot, the outbound-approval rule, the router and the watcher, each security-relevant check was
**deliberately broken** and the suite confirmed to fail: 33 distinct breakages, all caught. Two initially survived, and each
exposed a real weakness (two read-only layers masking each other; a dead check no test could reach) — fixed. This was done
by hand, not with a mutation-testing framework, and only for those four areas.

## Not defended

Malware or another user on your Mac · a malicious MCP server you enable · Groq or Google being compromised · someone
reading `data/` off an unencrypted disk · a model that is persuaded to do something a `safe`-tier tool allows (that is why
`safe` is reserved for reversible, local actions).
