// The stream renderer — Vesper's actual life, styled per the design system:
//   • voice lines  → Fraunces italic, typewriter reveal (skippable on click)
//   • plan traces  → dim mono "▸" steps, fade-up, collapse to "▸ N steps"
//   • observations → champagne gold (the ONLY gold text in the body)
//   • briefings    → restrained block: serif summary + mono data lines
//   • confirmations→ inline card, gold approve / hairline deny, 120s hairline
//   • idle         → nearly empty; just the last thing Vesper said, faded

const TYPE_MS = 30; // typewriter per character
const IDLE_MS = 9000; // fall idle after this much quiet
const MAX_NODES = 40; // prune the stream beyond this
const CONFIRM_MS = 120_000; // Guardian confirmation timeout
const ECHO_MS = 20_000; // window in which an identical reply is a replayed echo

type ConfirmFn = (requestId: string, approved: boolean) => void;

interface TraceGroup {
  el: HTMLElement;
  steps: HTMLElement;
  pending: HTMLElement[];
  count: number;
}

// Humanized verbs for the film-credit trace. Unknown tools fall back to their
// name with underscores relaxed.
const VERBS: Record<string, string> = {
  get_current_time: "checking the time",
  get_daily_briefing: "assembling your briefing",
  plan_my_day: "planning your day",
  close_app: "closing an app",
  open_app: "opening an app",
  lock_screen: "locking the screen",
  search_web: "searching the web",
  list_unread: "scanning the inbox",
  get_system_info: "reading system state",
  forget_memory: "forgetting on request",
};

function oneLine(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}
function trunc(s: string, n: number): string {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}
function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// A briefing should render as a block regardless of who asked for it (the HUD,
// the scheduler, or another client), so recognize it by shape: multi-sentence
// and covering both inbox and calendar.
function looksLikeBriefing(text: string): boolean {
  if (splitSentences(text).length < 2) return false;
  const t = text.toLowerCase();
  return /(email|inbox|unread)/.test(t) && /(calendar|meeting|schedule|event)/.test(t);
}

function stepLabel(tool: string, args: any): string {
  if (tool === "close_app" && args?.app_name) return `closing ${args.app_name}`;
  if (tool === "open_app" && args?.app_name) return `opening ${args.app_name}`;
  if (tool === "search_web" && args?.query) return `searching “${trunc(String(args.query), 22)}”`;
  return VERBS[tool] ?? tool.replace(/_/g, " ");
}

function resultSummary(msg: any): string {
  if (!msg.success) return msg.error ? trunc(oneLine(String(msg.error)), 40) : "failed";
  const r = oneLine(String(msg.result ?? ""));
  if (r) return trunc(r, 46);
  const ms = Number(msg.latency_ms ?? 0);
  return ms ? `${Math.round(ms)}ms` : "done";
}

export class Stream {
  private activeTrace: TraceGroup | null = null;
  private typing: { finish: () => void } | null = null;
  private idleTimer: number | undefined;
  private greetingShown = false;
  private expectingBriefing = false;
  private lastVoiceText = "";
  private lastVoiceAt = 0;

  constructor(
    private el: HTMLElement,
    private panel: HTMLElement,
    private onConfirm: ConfirmFn,
  ) {
    // Click anywhere completes an in-progress typewriter reveal.
    document.addEventListener("click", () => this.completeTyping());
  }

  /** Mark the next reply as a briefing (user typed "brief…"). */
  expectBriefing(): void {
    this.expectingBriefing = true;
  }

  handle(msg: any): void {
    switch (msg?.type) {
      case "snapshot":
        if (msg.greeting && !this.greetingShown) {
          this.greetingShown = true;
          this.voiceLine(msg.greeting);
          this.scheduleIdle();
        }
        break;
      case "reply":
        this.onReply(String(msg.text ?? ""));
        break;
      case "plan":
        // PlanCreatedEvent is a SUMMARY the Planner emits when the turn ends
        // (orchestrator/planner.py calls _emit_plan_trace on every exit path),
        // so it arrives AFTER the tool events, not before. Treating it as
        // "start a trace" collapsed the real group and left a spurious empty
        // one behind for the next reply to clean up. The trace is created by
        // the first tool_started instead; here it just closes the group.
        this.collapseTrace();
        break;
      case "tool_started":
        this.stepStart(msg);
        break;
      case "tool_finished":
        this.stepFinish(msg);
        break;
      case "observation":
        this.observation(String(msg.detail ?? ""));
        break;
      case "confirmation_requested":
        this.confirmCard(msg);
        break;
      case "briefing_requested":
        this.expectingBriefing = true;
        break;
    }
  }

  // ----------------------------- plumbing --------------------------------
  private append(node: HTMLElement): void {
    this.wake();
    node.classList.add("entry");
    this.el.appendChild(node);
    requestAnimationFrame(() => node.classList.add("in"));
    while (this.el.children.length > MAX_NODES) {
      this.el.removeChild(this.el.firstChild!);
    }
    this.scrollToEnd();
  }

  private scrollToEnd(): void {
    this.el.scrollTop = this.el.scrollHeight;
  }

  private wake(): void {
    this.panel.classList.remove("idle");
    if (this.idleTimer !== undefined) {
      clearTimeout(this.idleTimer);
      this.idleTimer = undefined;
    }
  }

  private scheduleIdle(): void {
    this.wake();
    this.idleTimer = window.setTimeout(() => this.panel.classList.add("idle"), IDLE_MS);
  }

  // ------------------------------ traces ---------------------------------
  private startTrace(): void {
    this.collapseTrace();
    const el = document.createElement("div");
    el.className = "trace";
    const steps = document.createElement("div");
    steps.className = "trace-steps";
    el.appendChild(steps);
    this.activeTrace = { el, steps, pending: [], count: 0 };
    this.append(el);
  }

  private ensureTrace(): TraceGroup {
    if (!this.activeTrace) this.startTrace();
    return this.activeTrace!;
  }

  private stepStart(msg: any): void {
    const t = this.ensureTrace();
    const step = document.createElement("div");
    step.className = "step";
    step.dataset.tool = String(msg.tool ?? "");
    step.textContent = `▸ ${stepLabel(msg.tool, msg.arguments)}…`;
    t.steps.appendChild(step);
    requestAnimationFrame(() => step.classList.add("in"));
    t.pending.push(step);
    t.count++;
    this.wake();
    this.scrollToEnd();
  }

  private stepFinish(msg: any): void {
    if (!this.activeTrace) return;
    const t = this.activeTrace;
    const tool = String(msg.tool ?? "");
    let idx = t.pending.findIndex((s) => s.dataset.tool === tool);
    if (idx < 0) idx = 0;
    const step = t.pending.splice(idx, 1)[0];
    if (!step) return;
    const summary = resultSummary(msg);
    step.textContent = `▸ ${stepLabel(tool, {})}${summary ? " — " + summary : ""}`;
    step.classList.toggle("fail", !msg.success);
    this.scrollToEnd();
  }

  private collapseTrace(): void {
    const t = this.activeTrace;
    if (!t) return;
    this.activeTrace = null;
    if (t.count === 0) {
      t.el.remove();
      return;
    }
    t.el.classList.add("folded");
    const toggle = document.createElement("button");
    toggle.className = "trace-toggle";
    const word = t.count === 1 ? "step" : "steps";
    toggle.textContent = `▸ ${t.count} ${word}`;
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      const open = t.el.classList.toggle("open");
      toggle.textContent = `${open ? "▾" : "▸"} ${t.count} ${word}`;
    });
    t.el.insertBefore(toggle, t.steps);
  }

  // ------------------------------ replies --------------------------------
  private onReply(text: string): void {
    this.collapseTrace();

    // The gateway replays the session greeting in its connect snapshot, and a
    // wake fires a fresh greeting moments later — on the wake shot that read
    // as Vesper saying "Good evening, Sir." twice in a row. Collapse an exact
    // repeat that lands within the echo window onto the existing line instead
    // of appending a duplicate. Deliberately narrow: a genuine repeat later in
    // the session still gets its own line.
    if (text && text === this.lastVoiceText && Date.now() - this.lastVoiceAt < ECHO_MS) {
      this.scheduleIdle();
      return;
    }
    const briefing = this.expectingBriefing || looksLikeBriefing(text);
    this.expectingBriefing = false;
    if (briefing) {
      this.briefing(text);
    } else {
      this.voiceLine(text);
    }
    this.scheduleIdle();
  }

  private voiceLine(text: string): void {
    this.completeTyping();
    this.lastVoiceText = text;
    this.lastVoiceAt = Date.now();
    const line = document.createElement("div");
    line.className = "line voice";
    this.append(line);

    const full = text;
    let i = 0;
    let interval: number | undefined = window.setInterval(() => {
      i++;
      line.textContent = full.slice(0, i);
      this.scrollToEnd();
      if (i >= full.length) {
        if (interval !== undefined) clearInterval(interval);
        interval = undefined;
        this.typing = null;
      }
    }, TYPE_MS);

    this.typing = {
      finish: () => {
        if (interval !== undefined) clearInterval(interval);
        interval = undefined;
        line.textContent = full;
        this.typing = null;
        this.scrollToEnd();
      },
    };
  }

  private completeTyping(): void {
    if (this.typing) this.typing.finish();
  }

  // ----------------------------- briefing --------------------------------
  private briefing(text: string): void {
    const block = document.createElement("div");
    block.className = "briefing";
    const parts = splitSentences(text);
    const summary = document.createElement("div");
    summary.className = "briefing-summary";
    summary.textContent = parts[0] ?? text;
    block.appendChild(summary);
    for (const line of parts.slice(1)) {
      const l = document.createElement("div");
      l.className = "briefing-line";
      l.textContent = line;
      block.appendChild(l);
    }
    this.append(block);
  }

  // --------------------------- observations ------------------------------
  private observation(detail: string): void {
    const line = document.createElement("div");
    line.className = "line obs";
    line.textContent = detail;
    this.append(line);
    this.scheduleIdle();
  }

  // -------------------------- confirmation card --------------------------
  private confirmCard(msg: any): void {
    const requestId = String(msg.request_id ?? "");
    const card = document.createElement("div");
    card.className = "confirm-card";

    const summary = document.createElement("div");
    summary.className = "confirm-summary";
    summary.textContent = msg.summary || `Approve ${msg.tool_name ?? "this action"}?`;

    const actions = document.createElement("div");
    actions.className = "confirm-actions";
    const approve = document.createElement("button");
    approve.className = "cbtn approve";
    approve.textContent = "Approve";
    const deny = document.createElement("button");
    deny.className = "cbtn deny";
    deny.textContent = "Deny";
    actions.append(approve, deny);

    const expiry = document.createElement("div");
    expiry.className = "confirm-expiry";
    const bar = document.createElement("span");
    bar.className = "expiry-bar";
    expiry.appendChild(bar);

    card.append(summary, actions, expiry);
    this.append(card);

    // Deplete the hairline over the Guardian's 120s window.
    requestAnimationFrame(() => {
      bar.style.transition = `width ${CONFIRM_MS}ms linear`;
      bar.style.width = "0%";
    });

    let resolved = false;
    const settle = (state: "approved" | "denied" | "expired", send?: boolean): void => {
      if (resolved) return;
      resolved = true;
      card.classList.add(state);
      bar.style.transition = "none";
      approve.disabled = true;
      deny.disabled = true;
      const status = document.createElement("div");
      status.className = "confirm-status";
      status.textContent = state === "approved" ? "✓ approved" : state === "denied" ? "✕ denied" : "expired";
      actions.replaceWith(status);
      if (send !== false && (state === "approved" || state === "denied")) {
        this.onConfirm(requestId, state === "approved");
      }
    };

    approve.addEventListener("click", (e) => {
      e.stopPropagation();
      settle("approved");
    });
    deny.addEventListener("click", (e) => {
      e.stopPropagation();
      settle("denied");
    });
    window.setTimeout(() => settle("expired", false), CONFIRM_MS);
  }
}
