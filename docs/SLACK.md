# Slack in the briefing (read-only)

Unread **DMs, group DMs and @mentions** show up in "good morning", ranked by the same deterministic scorer as mail.
Slack is **read-only** here: the collector can call six Slack read methods and nothing else — no posting, no
reactions, no mark-as-read. It is **off until you give it a token**; without one it reports "not configured" and the
briefing is exactly what it was.

Not verified against a live workspace yet — see the checklist at the end.

## What it does

| Slack | Becomes | Notes |
|---|---|---|
| A DM or group DM from someone else, newer than Slack's own "read up to here" marker | Item `source="slack"`, kind `dm` / `group_dm` | read it in the Slack app and it drops out at the next run |
| A message in a public channel you are in that `@mentions` you | kind `mention` | found with `search.messages`; needs `search:read` |
| Ordinary channel chatter, your own messages, joins/edits, muted channels | ignored | channels are never listed or read, only searched for mentions |

Each item keeps: sender display name (≤32 chars), a **hard-truncated** snippet (`slack.snippet_chars`, 160), a
timestamp, thread/reply flags. The briefing says `Slack: 3 unread DMs, 1 mention` and **names only the top
`builder.max_slack_items` (2)**. Everything else is counted, never named.

**Cursors.** Each run asks Slack only for messages newer than `max(last successful run − 2 min, your read marker)`
(first run: `slack.lookback_days`, 2). The cursor advances only on success. Read-state is reconciled every run: a
message you read in Slack on any device stops being "live" and drops out of the cache.

## Create the app (once, ~3 minutes)

1. Go to <https://api.slack.com/apps> → **Create New App** → **From a manifest**, pick your workspace, paste:

   ```yaml
   display_information:
     name: Vesper briefing (read-only)
   oauth_config:
     scopes:
       user:                 # USER scopes only — no bot user, no bot scopes
         - im:history
         - im:read
         - mpim:history
         - mpim:read
         - channels:history
         - channels:read
         - users:read
         - search:read
   settings:
     org_deploy_enabled: false
     socket_mode_enabled: false
     token_rotation_enabled: false
   ```
2. **Install to Workspace** → allow. Copy the **User OAuth Token** (starts `xoxp-`). Not the bot token (there is none).
3. Store it where Vesper looks (it checks `$VESPER_SLACK_USER_TOKEN` first, then the Keychain):

   ```bash
   security add-generic-password -s vesper-slack-user -a user-token -w     # prompts for the token; nothing in shell history
   ```
   A token in `.env` also works (`VESPER_SLACK_USER_TOKEN=xoxp-…`); `.env` is git-ignored. Never commit it.
4. Restart `vesper up` (or wait ≤ 1 minute: a "not configured" answer is rechecked every 60 s).

### The exact scopes, and why

| Scope | Used for | Your list | |
|---|---|---|---|
| `im:history` | read DM messages | ✔ | |
| `mpim:history` | read group-DM messages | ✔ | |
| `channels:history` | read public channels you are in | ✔ | only the messages `search.messages` returns |
| `users:read` | sender display name, bot/app flag | ✔ | |
| `search:read` | find @mentions in channels | ✔ | **optional**: without it you get DMs only (the briefing logs why) |
| `im:read`, `mpim:read` | list your DMs / group DMs; read each one's "last read" marker | **added** | `*:history` alone cannot enumerate conversations, and without the marker everything would look unread |
| `channels:read` | the "last read" marker of a channel that mentioned you | **added** | |

All are read scopes. Nothing with `write`, `chat:`, `reactions:`, `*:write` or `admin` is requested, and the code
could not use them anyway (below). Private channels (`groups:*`) are deliberately **not** requested.

### If your workspace requires admin approval

Slack shows **Request to Install** instead of **Install**. Send the request with a one-line reason; the admin gets
the scope list above and approves or denies it in *Manage apps*. Things that help it pass: user scopes only, no bot,
no write scopes, and the app is private to you (never distributed). While it is pending nothing happens: Vesper stays
"not configured" and the briefing is unchanged.

If the admin will only approve less, drop scopes in the manifest and set the matching config:
* no `search:read` → `slack.mentions_via_search: false` (DMs and group DMs only);
* keep `im:*` and `users:read` as the minimum for DMs; group DMs also need both `mpim:*` scopes (without them the run
  fails visibly with `missing_scope` and the source shows STALE rather than silently skipping them).

If the workspace forbids custom apps entirely, there is no read-only alternative that avoids a token; leave Slack off.

## Config (`config/briefing.yaml`, personal values in the git-ignored `config/briefing.local.yaml`)

```yaml
slack:
  muted_channels: ["random", "C01234ABCDE"]   # Slack does not expose its own "muted" flag via the API: list them here
  allowlisted_senders: ["Github", "U01234ABCDE"]   # bots/apps you DO want (user/bot ids or display names)
  mentions_via_search: true
```
Weights (all in `weights:` with a comment each): `slack_dm_human` +25, `slack_mention` +20, `slack_reply_to_mine` +10,
`slack_decay_per_day` −5 (floor `slack_decay_floor` −30) for unread older than `fresh_hours` (24), `slack_bot_penalty`
−30 unless allowlisted. The existing `fresh_bonus`, `urgency_keywords` and `injection_suspected` apply as for mail.
`vesper briefing --explain` shows every point.

## Read-only, enforced

`briefing/slack_api.py` is the only code that talks to Slack. `SlackReadOnlyClient.call(method)` raises
`ReadOnlyViolation` for anything not in

```
SLACK_READ_METHODS = auth.test · users.conversations · conversations.info · conversations.history · users.info · search.messages
```

*before* a request is built; the collector holds a client, not an SDK. Tests (`tests/test_slack_readonly.py`) assert the
allow-list contents, a refusal for ~50 mutating methods (`chat.postMessage`, `reactions.add`, `conversations.mark`,
`conversations.join`…) with **no request made**, malformed/smuggled method names, a source scan of both files, and that
the only transport call and HTTP POST in the package are the guarded ones. Reading through the API does not mark
anything read in Slack and does not tell senders you looked.

## Hostile text

All Slack text is `third_party` — as hostile as an email subject. Same layers as mail (see BRIEFING.md → Hostile
content): markup/links/addresses stripped and the snippet hard-truncated at collection; shell substitutions defused
(`$(`→`$ (`); text that reads like instructions to an AI, or carries a shell payload, is flagged `hostile`, **stores no
snippet at all**, is scored −40, and is shown as `[withheld]` (its sender is not named either); every string is a
quoted JSON literal in a block that says "DATA"; and `get_daily_briefing` is registered `untrusted_output`, so the
Guardian's tainted-input rule applies to whatever the model does next — the real backstop.

## Limits (honest list)

* **Muted channels** can't be read from the API; you list them in config. (Channels are only ever searched for
  mentions, so muting matters only for noisy `@here`/`@channel`-style mentions.)
* **Thread replies to your messages** are only seen when Slack includes them in a conversation's history or a search
  (broadcast replies, replies in DM threads that surface). A quiet in-thread reply under your message in a channel is
  invisible, so the +10 applies less often than you'd hope.
* DMs are examined up to `slack.max_conversations` (60) per run; Slack doesn't order them by activity, so with more
  than that, some are not checked. Rate limits (`ratelimited`) fail that run visibly and retry at the next one.
* Search can lag a few minutes behind a new message; `search.messages` only searches what your token can see.
* Private channels, files and reactions are out of scope.

## Manual checklist (needs your real workspace — nothing here has been run against one)

1. Create the app and store the token as above. `vesper briefing --refresh --explain --redact` prints Slack rows as
   `<sender:N> — <subject:N>`; the run line says `slack: ok (N item(s) fetched)`.
2. Have someone DM you (first check your unread badge stays unread after a `--refresh`: API reads should not mark
   anything read — if they do, stop and tell me): next `--refresh` shows `Slack: 1 unread DM`. Read it in Slack: next `--refresh` drops it.
3. Get @mentioned in a public channel: it appears as a mention. If it doesn't, check `search:read` (the log says
   `slack mentions unavailable: missing_scope`).
4. Mute a noisy channel in `briefing.local.yaml`; its mentions disappear.
5. Revoke the token in Slack: the next run shows `slack STALE … invalid_auth` while Gmail and Calendar keep working.
6. Remove the token: Slack vanishes from the briefing entirely (no "never synced", no staleness note).
7. Confirm in *Slack → Apps → your app* that only the user scopes above are granted.
8. `say "good morning"`: the spoken script names at most two Slack senders and ends with its usual question.
