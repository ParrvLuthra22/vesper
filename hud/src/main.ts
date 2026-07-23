// Vesper HUD — frontend (PV2).
//
// Bootstraps the live clock, the gateway link, and the stream renderer, then
// wires the light command input. The HUD is display-first: the input is for
// short commands and confirmations, and replies flow into the voice-line area.

import { GatewayLink, type ConnState } from "./gateway";
import { Stream } from "./render";

const panel = document.getElementById("panel")!;
const star = document.getElementById("star")!;
const clockEl = document.getElementById("clock")!;
const streamEl = document.getElementById("stream")!;
const composer = document.getElementById("composer") as HTMLFormElement;
const input = document.getElementById("input") as HTMLInputElement;

const prefersReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let listeningTimer: number | undefined;

// --------------------------- The wake reveal ------------------------------
// Wake word fired → the star flares, the panel wakes, and (via the greeting
// reply that follows) Vesper speaks + types. Reduced motion degrades to a
// quiet un-idle with no bloom.
function wakeFlow(animate: boolean): void {
  panel.classList.remove("idle");
  if (animate && !prefersReducedMotion()) {
    star.classList.remove("flaring");
    void star.offsetWidth; // restart the animation
    star.classList.add("flaring");
    panel.classList.add("waking");
    window.setTimeout(() => star.classList.remove("flaring"), 700);
    window.setTimeout(() => panel.classList.remove("waking"), 1500);
  }
  // Then the listening indicator: focus the input, show it's listening.
  panel.classList.add("listening");
  input.placeholder = "listening…";
  try {
    input.focus();
  } catch {
    /* focus:false windows may refuse */
  }
  if (listeningTimer !== undefined) clearTimeout(listeningTimer);
  listeningTimer = window.setTimeout(() => {
    panel.classList.remove("listening");
    input.placeholder = "…";
  }, 9000);
}

// --------------------------------- Clock ----------------------------------
function tickClock(): void {
  const now = new Date();
  let h = now.getHours();
  const suffix = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  const pad = (n: number) => n.toString().padStart(2, "0");
  clockEl.textContent = `${pad(h)}:${pad(now.getMinutes())}:${pad(now.getSeconds())} ${suffix}`;
}
tickClock();
setInterval(tickClock, 1000);

// ------------------------------ Wiring ------------------------------------
const link = new GatewayLink(
  (msg) => {
    if (msg && msg.type === "wake") {
      wakeFlow(msg.animate !== false);
      return;
    }
    stream.handle(msg);
  },
  (state, port) => setConn(state, port),
);

const stream = new Stream(streamEl, panel, (id, approved) => {
  void link.confirm(id, approved);
});

// Dev/test hook: drive the renderer directly (e.g. from a browser preview or
// an automated check) without needing the live gateway. Harmless in prod.
(window as unknown as { __hud?: unknown }).__hud = {
  handle: (m: unknown) => stream.handle(m),
  briefing: () => stream.expectBriefing(),
  connected: (on: boolean) => setConn(on ? "connected" : "disconnected", "8760"),
  wake: (animate = true) => wakeFlow(animate),
};

function setConn(state: ConnState, _port: string): void {
  panel.classList.remove("connected", "disconnected");
  if (state === "connected") panel.classList.add("connected");
  if (state === "disconnected") panel.classList.add("disconnected");
  input.placeholder = state === "connected" ? "…" : "offline";
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  if (/^\s*brief/i.test(text)) stream.expectBriefing();
  const sent = link.sendMessage(text);
  if (sent) input.value = "";
});

void link.start();
