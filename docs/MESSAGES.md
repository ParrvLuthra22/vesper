# Messages (iMessage / SMS) in the briefing — local and read-only

Unread, **unanswered** texts show up in "good morning": `Messages: 3 unread` in the planner block, "On Messages, 3
unread, the top one from Alice." in the spoken script, and a row per message in `vesper briefing --explain`.
Everything happens on this Mac. **Nothing is sent, marked read, deleted or written** — to Messages or anywhere else.

It is **off by default.** Two things must be true before anything is read: you set `messages.enabled: true`, and
macOS lets your terminal open `~/Library/Messages/chat.db` (Full Disk Access). Without either, the briefing is
exactly what it was; no source line, no staleness note.

## Setup (once, ~2 minutes)

1. **System Settings → Privacy & Security → Full Disk Access** → add your terminal app (Terminal, iTerm, Ghostty…)
   and switch it on. Quit and reopen the terminal.
2. `vesper doctor` — the **Messages: Full Disk Access** line should say *chat.db can be opened*. (The check only
   tries to open the file: no byte is read, no prompt is shown.)
3. In `config/briefing.local.yaml` (git-ignored):
   ```yaml
   messages:
     enabled: true
     muted_chats: ["Family group"]      # optional: chat_identifier or group name
     allowlisted_senders: ["AMAZON"]    # optional: short codes you DO want treated as people-grade
   ```
4. `vesper briefing --refresh --explain --redact` — redacted output: placeholders, never real names or text.

## How it reads chat.db

`chat.db` (and its `-wal`/`-shm`) is **copied** into a private `0700` temp directory, the copy is opened read-only
(`mode=ro` + `PRAGMA query_only`), queried, and the whole directory is deleted — also when a query fails. The original
is only ever opened for reading by the copy. A test asserts the snapshot connection rejects `INSERT/UPDATE/DELETE/
CREATE/DROP`, that the original file's hash and mtime are unchanged, and that no temp copy is left behind.

## What becomes an item

| In chat.db | Becomes | Notes |
|---|---|---|
| An incoming 1:1 or group message that is unread and newer than my last message in that chat | `source="messages"` item | the thread is "pending" only while I have not replied |
| `text` NULL, text in `attributedBody` (macOS 13+) | decoded from the typedstream | an unreadable blob is skipped, never guessed at |
| Tapbacks/reactions/stickers, voice notes, attachment-only messages, system rows (renames, joins), my own messages | ignored | |
| Chats in `messages.muted_chats` | ignored | macOS does not expose per-chat mute in `chat.db`, so mute here |

Cursor: the highest message `ROWID` seen. Read-state is reconciled every run from the same snapshot, so a message you
read (or answered) on any device drops out of the cache.

Sender: a contact name if the local Address Book has the number/email (read from a private copy too; turn off with
`messages.contacts: false`), otherwise the number or email. **Numbers are never spoken** ("an unknown number").

## Scoring (all weights in `config/briefing.yaml`)

| Signal | Default |
|---|---|
| unread 1:1 message from a person (phone/email) | +25 `messages_dm_human` |
| group chat message | +10 `messages_group_human` |
| unread and older than `messages.unanswered_hours` (3) | +5 `messages_unanswered` |
| urgency words (the shared list) | as for mail |
| short code / alphanumeric sender ("HDFCBK", `urn:biz:`) | −30 `messages_automated_penalty` (urgency words ignored) |
| one-time-code message ("123456 is your code") | −40 `messages_otp_penalty` |
| fresh / decay | as for mail; decay `messages_decay_per_day` −5, floor −30 |
| reads like instructions to an AI | −40 and no snippet |

## One-time codes

Every run of 4+ digits (allowing single `-`, space or `.` between groups) is replaced by `(code)` **before** the
snippet is truncated or stored — so a code never reaches the cache, the planner block, the spoken script, `--explain`,
or a log, and a truncation can never leave half of one. This also masks phone numbers and amounts inside message
text; that is deliberate.

## Hostile text

Same defenses as mail and Slack (`briefing/sanitize.py`, `briefing/slack.hostile_text`): hard truncation, URLs and
addresses replaced, shell substitutions flagged; text addressed to an AI stores **no snippet**, is demoted, is never
named, and is reported only as "N messages filtered as unsafe". The briefing tool is `untrusted_output`, so the
Guardian's taint rule applies to whatever the model does after reading it.

## Redaction

`vesper briefing --redact` replaces every Messages sender, number and text with placeholders and withholds all output
if any stored value would be printed. Tested with a synthetic canary (fake names, numbers, group name and text),
including a simulated output-layout change.

## Known limits

- **Not verified against a real `chat.db` in this repo** — all tests use a synthetic database with the columns the
  collector reads (`message`, `chat`, `chat_message_join`, `handle`). Apple can change the schema between macOS
  releases; if a column is missing the run fails (visible in `vesper doctor` as a stale source) instead of guessing.
- `attributedBody` decoding handles the plain `NSString` typedstream layout (short, 2-byte and 4-byte lengths); other
  layouts are skipped.
- "Muted" is configuration, not read from macOS.
- Read receipts: Vesper never marks anything read, so a message stays "unread" in Messages until you read it there.
