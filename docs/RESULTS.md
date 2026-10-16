# Results

Only numbers that were **measured** appear here, each with where it came from and when. Anything that has not been
measured is in the last section, marked as such. "This repo" means a run of the code in this repository; older
measurements are cited to the document that recorded them and were **not** re-run when this page was written.

## Test suite (this repo)

`.venv/bin/python -m pytest` — **1,532 passed, 8 deselected** (as of the commit that added this page) (the 8 deselected are the `integration` tests that need a live LLM and network;
`pytest.ini` skips them by default).

| Step of the October work | Passing tests | Added |
|---|---|---|
| Start of the run (briefing, Slack, Telegram, launcher, redaction fixes) | 1,294 | – |
| + Messages collector, doctor check | 1,371 | 77 |
| + outbound-approval rule, `gmail_create_draft` | 1,435 | 64 |
| + deterministic skill router | 1,470 | 35 |
| + proactive watcher | 1,525 | 55 |
| + outbound MCP tools, docs link checks, unknown-subcommand guard | 1,532 | 7 |

## Security checks that were validated by breaking them (this repo)

Hand-run mutation testing of the security-relevant checks added in the October work (not an automated framework):

| Area | Distinct breakages tried | Caught |
|---|---|---|
| Messages read-only snapshot (masking, `mode=ro`, `query_only`, both, hostile snippet, tapbacks, answered threads) | 7 | 7 |
| Outbound-approval rule and draft tool (guardian check, any-source-is-local, channel door, outbound-may-be-safe, recipient check, draft→dangerous, draft not Mac-only, source scan) | 8 | 8 |
| Deterministic router (anchor, ambiguity, `close_app` floor, never-close list, Guardian call, volume range, greeting hour) | 7 | 7 |
| Proactive watcher (DND, quiet hours, budget, cooldown, dedupe, payload leak, hostile sender, suspicious items, stale re-arm, day roll, midnight wrap) | 11 | 11 |
| **Total** | **33** | **33** |

Two breakages survived at first and were fixed by strengthening the tests (the two read-only layers masked each other; one
"check" turned out to be dead code). A no-op control mutant correctly survived.

## Redaction canaries (this repo; counts only)

* **Synthetic inbox** (invented names, a channel, message text and a fake `@mention`, 19 canary strings): after every
  `--redact` command (`--explain`, `--json`, `--known`, `--rules`, both `--mark` forms, plus the two unsupported views),
  **0 of 19** canaries survived. A simulated output-layout change that printed raw values was **withheld** by the guard
  (exit 3, nothing printed); the same formatter without `--redact` leaked 13 canaries, which is what proves the probe bites.
* **A scratch copy of the real briefing cache**, checked independently of the redactor against 115 stored full values,
  940 three-word windows, 48 name/address tokens and 27 known addresses/domains: **0 leaks** across `--explain`, `--json`,
  `--known`, `--rules` and `--mark`. Before the over-blocking fix, `--known` and `--mark` were withheld on that data
  (a false positive, not a leak); after it, both print and still leak 0.
* The real cache held 36 Gmail items and **no** Slack or Messages items, so Slack/Messages redaction was verified on
  synthetic data only.

## Router (this repo, `python tests/eval_router.py`, dev Mac)

156 hand-written utterances, clock pinned to 08:00, machine-independent. Full table and the list of known gaps:
[ROUTER.md](ROUTER.md).

| Metric | Result |
|---|---|
| Handled without the LLM | 97 / 156 = 62.2 % |
| Precision of what it handled | 100 % |
| Precision on the 59 dangerous/ambiguous utterances | 100 % (0 wrongly claimed) |
| Recall on the 97 it was built for | 100 % (optimistic — written alongside the patterns) |
| **Recall on 39 unseen phrasings** | **53.8 %**, 0 wrong routes |
| Median router latency | 11 µs per utterance |

## Briefing and planning cost

| Number | Source | Measured |
|---|---|---|
| Briefing context for the planner: **206 tokens vs 3,159** on the same real inbox (~20 raw messages); the "good morning" fast path skips the LLM entirely | [ARCHITECTURE.md](ARCHITECTURE.md) §8c | earlier, on the real inbox; not re-run for this page |
| The briefing block respects its cap: 505 tokens / cap 600 with mail, Slack, Messages, 2 meetings and an unsafe-item line present | `vesper briefing --cache-only` on the **synthetic** demo cache (`scripts/demo_seed.py`) | this repo |
| Planner prompt: **1,849 → 851 `tokens_in` (−54 %)** from per-turn tool filtering, against real Groq calls | [ARCHITECTURE.md](ARCHITECTURE.md) §7 | earlier |
| Semantic memory: MiniLM **5/5** vs hash embeddings **2/5** top-1 on a benchmark of 5 lexically-disjoint query/memory pairs (a small sample) | [ARCHITECTURE.md](ARCHITECTURE.md) §6 | earlier |

## Footprint and latency on the 8 GB M3 (from [RESOURCES.md](RESOURCES.md), measured 2026-10-02)

Measured with `phys_footprint` on the real stack under heavy existing memory pressure (8 GB swap in use before Vesper started).

| Component | Idle | After a turn |
|---|---|---|
| Gateway (Brain, planner, 2 MCP servers) | 505 MB | 695 MB |
| Voice output (Kokoro) | 35 MB (≈390–415 MB with `preload: true`, now the default) | 494 MB |
| HUD (Tauri) | 62 MB | 134 MB |
| Voice input (wake word + VAD) | 150 MB | 151 MB (+263 MB once Whisper loads) |
| **Whole stack** | **752 MB** (≈1.1 GB with Kokoro preloaded) | **≈1.5 GB**, ≈1.7–1.9 GB at worst |

| Step | Time |
|---|---|
| STT, 4.8 s utterance, `base.en` int8 | 3.1 s first (includes model load), **0.41 s warm** (0.09× realtime) |
| Planner call (Groq `gpt-oss-120b`) | 0.5–1.4 s each; 2–3 calls per tool turn; 4.5 s for the first call after boot |
| Local rescue `llama3.2:3b` Q4 | +2.3 GB resident, 6.6 s cold load, 37 tok/s |
| Kokoro synthesis | 0.55× realtime on an M3; model load 1.9 s ([`voice/output/README.md`](../voice/output/README.md)) |

## Routing eval (Vantage, a live run against Groq)

`orchestrator_v1_smoke`, 10 scenarios: **70 % (7/10)** on 2026-09-23 against a committed baseline of 90 % (9/10); the gap was
model variance on an unchanged prompt, not a code change ([README](../README.md#evaluation)). A single run — treat it as
a smoke test, not a benchmark.

## Not measured (stated so nobody has to guess)

* **The full voice round trip on a microphone.** An acoustic loop (the Mac's speakers saying "Hey Jarvis" into its own mic)
  did not trigger the wake word; only the stages were measured separately (above).
* **Anything against a live Slack workspace, a real `chat.db`, or a real Gmail draft.** The Slack collector, the Messages
  collector and `gmail_create_draft` were built and tested against synthetic fixtures and mocks.
* **A real macOS banner or HUD call-out from the proactive watcher** (tests use a fake notifier and bus).
* **Token or money saved by the router.** Each handled utterance avoids one planner turn; the saving per turn depends on the
  prompt size and was not measured.
* **Router accuracy on real speech-to-text output** (the eval is typed text) or beyond the 39 unseen phrasings.
* **Groq's behaviour at scale.** The free tier's 8,000 tokens/minute limit is the practical ceiling; its effect on a long
  session was observed during the 2026-10-02 audit (two of six turns in a burst failed with "I'm having trouble thinking right now") but not quantified in a controlled test.
