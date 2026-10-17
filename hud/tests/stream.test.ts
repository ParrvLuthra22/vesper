// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Stream } from "../src/render";

let el: HTMLElement;
let panel: HTMLElement;
let stream: Stream;

beforeEach(() => {
  vi.useFakeTimers();
  document.body.innerHTML = '<div class="panel" id="panel"><main class="stream" id="stream"></main></div>';
  el = document.getElementById("stream")!;
  panel = document.getElementById("panel")!;
  Object.defineProperty(el, "scrollHeight", { configurable: true, get: () => 9999 });
  stream = new Stream(el, panel, () => {});
});

const REPLIES = [
  "First reply. It is long enough to wrap over several lines in a narrow panel, which is what squeezed it before.",
  "Second reply, also long, with more words than the panel is tall so the stream has to scroll to keep up.",
  "Third reply arrives while the second is still being typed out.",
];

describe("messages are separate blocks in normal flow", () => {
  it("three replies in a row get three separate sibling blocks", () => {
    for (const r of REPLIES) stream.handle({ type: "reply", text: r });
    const blocks = Array.from(el.children) as HTMLElement[];
    expect(blocks).toHaveLength(3);
    expect(new Set(blocks).size).toBe(3);
    for (const b of blocks) {
      expect(b.parentElement).toBe(el);
      expect(b.classList.contains("line") && b.classList.contains("voice")).toBe(true);
      expect(b.style.position).toBe("");
      expect(b.style.top + b.style.left + b.style.bottom + b.style.right).toBe("");
    }
  });

  it("a streaming (typewriter) message updates in place and never creates or overwrites another block", () => {
    stream.handle({ type: "reply", text: REPLIES[0] });
    const first = el.children[0] as HTMLElement;
    vi.advanceTimersByTime(30 * 10);
    expect(el.children).toHaveLength(1);
    expect(first.textContent).toBe(REPLIES[0].slice(0, 10));
    stream.handle({ type: "reply", text: REPLIES[1] }); // the next reply completes the previous one first
    expect(el.children).toHaveLength(2);
    expect(first.textContent).toBe(REPLIES[0]); // untouched by the second reply
    vi.advanceTimersByTime(30 * 5);
    expect(el.children[1].textContent).toBe(REPLIES[1].slice(0, 5));
    expect(first.textContent).toBe(REPLIES[0]);
  });

  it("scrolls to the newest message every time one arrives", () => {
    el.scrollTop = 0;
    stream.handle({ type: "reply", text: REPLIES[0] });
    expect(el.scrollTop).toBe(9999);
    el.scrollTop = 0;
    stream.handle({ type: "reply", text: REPLIES[1] });
    expect(el.scrollTop).toBe(9999);
  });

  it("a briefing, an observation and a trace are blocks too, in arrival order", () => {
    stream.handle({ type: "observation", detail: "Heads up" });
    stream.handle({ type: "tool_started", tool: "open_app", arguments: { app_name: "Notes" } });
    stream.handle({ type: "reply", text: "Good evening, Sir. You have one email and one meeting on your calendar." });
    const kinds = Array.from(el.children).map((c) => c.className.split(" ")[0]);
    expect(kinds).toEqual(["line", "trace", "briefing"]);
  });
});

describe("the stylesheet cannot make messages overlap", () => {
  const css = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].map((m) => ({ sel: m[1].trim(), body: m[2] }));

  it("no message class is absolutely or fixed positioned", () => {
    const MESSAGE = /(^|[\s,>])\.(entry|line|voice|obs|briefing|briefing-line|briefing-summary|trace|confirm-card)\b/;
    const bad = rules.filter((r) => MESSAGE.test(r.sel) && /position\s*:\s*(absolute|fixed)/.test(r.body));
    expect(bad.map((r) => r.sel)).toEqual([]);
  });

  it("stream children never shrink (flex: 0 0 auto) and nothing squeezes a voice line with min-height", () => {
    const direct = rules.find((r) => /^\.stream\s*>\s*\*$/.test(r.sel));
    expect(direct?.body).toMatch(/flex\s*:\s*0\s+0\s+auto/);
    const voice = rules.find((r) => r.sel === ".line.voice");
    expect(voice?.body ?? "").not.toMatch(/\bheight\s*:\s*\d+px/);
  });
});
