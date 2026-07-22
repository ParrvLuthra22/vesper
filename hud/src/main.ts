// Vesper HUD — frontend shell (PV1).
//
// Responsibilities this session: a live clock, and a resilient WebSocket to the
// gateway whose connection state drives the evening star (breathing when live,
// a dead ember when not). Incoming events are only printed as a stream of their
// types — proof that data flows. Feature rendering is PV2.

import { invoke } from "@tauri-apps/api/core";

type Conn = "connecting" | "connected" | "disconnected";

const panel = document.querySelector<HTMLElement>(".panel")!;
const clockEl = document.getElementById("clock")!;
const connLabel = document.getElementById("conn-label")!;
const streamEl = document.getElementById("stream")!;

const MAX_STREAM = 8;
const BACKOFF_MIN_MS = 500;
const BACKOFF_MAX_MS = 10_000;

// --------------------------------- Clock ----------------------------------
function tickClock(): void {
  const now = new Date();
  let h = now.getHours();
  const m = now.getMinutes();
  const s = now.getSeconds();
  const suffix = h >= 12 ? "PM" : "AM";
  h = h % 12 || 12;
  const pad = (n: number) => n.toString().padStart(2, "0");
  clockEl.textContent = `${pad(h)}:${pad(m)}:${pad(s)} ${suffix}`;
}
tickClock();
setInterval(tickClock, 1000);

// ---------------------------- Connection state ----------------------------
function setConn(state: Conn, detail = ""): void {
  panel.classList.remove("connected", "disconnected");
  if (state === "connected") panel.classList.add("connected");
  if (state === "disconnected") panel.classList.add("disconnected");
  const label =
    state === "connected"
      ? `connected${detail ? " · " + detail : ""}`
      : state === "disconnected"
        ? `offline${detail ? " · " + detail : ""}`
        : "connecting…";
  connLabel.textContent = label;
}

function pushEvent(type: string): void {
  const li = document.createElement("li");
  li.textContent = type;
  li.classList.add("fresh");
  streamEl.appendChild(li);
  // Only the newest entry keeps the "fresh" highlight.
  for (const el of Array.from(streamEl.children)) {
    if (el !== li) el.classList.remove("fresh");
  }
  while (streamEl.children.length > MAX_STREAM) {
    streamEl.removeChild(streamEl.firstChild!);
  }
}

// ------------------------------- Gateway link -----------------------------
async function resolveGateway(): Promise<{ wsUrl: string; port: string }> {
  // Rust owns config (file + env). Fall back to defaults if we're running
  // outside Tauri (e.g. a plain-browser preview).
  let url = "ws://127.0.0.1:8760/ws";
  let token = "";
  try {
    const cfg = await invoke<{ url: string; token: string }>("get_gateway_config");
    url = cfg.url || url;
    token = cfg.token || "";
  } catch {
    // not in Tauri — leave defaults; the socket will just ember out.
  }
  const port = url.match(/:(\d+)\//)?.[1] ?? "";
  const wsUrl = token ? `${url}?token=${encodeURIComponent(token)}` : url;
  return { wsUrl, port };
}

let backoff = BACKOFF_MIN_MS;
let reconnectTimer: number | undefined;

async function connect(): Promise<void> {
  const { wsUrl, port } = await resolveGateway();
  setConn("connecting");

  let ws: WebSocket;
  try {
    ws = new WebSocket(wsUrl);
  } catch {
    scheduleReconnect();
    return;
  }

  ws.onopen = () => {
    backoff = BACKOFF_MIN_MS; // reset on a good connection
    setConn("connected", port ? `:${port}` : "");
  };

  ws.onmessage = (ev) => {
    let type = "?";
    try {
      type = (JSON.parse(ev.data as string).type as string) ?? "?";
    } catch {
      /* ignore non-JSON frames */
    }
    pushEvent(type);
  };

  ws.onerror = () => {
    // onclose always follows; let it handle reconnect.
    try {
      ws.close();
    } catch {
      /* noop */
    }
  };

  ws.onclose = () => {
    setConn("disconnected");
    scheduleReconnect();
  };
}

function scheduleReconnect(): void {
  if (reconnectTimer !== undefined) return;
  // Exponential backoff with jitter, capped.
  const jitter = Math.random() * 0.3 * backoff;
  const delay = Math.min(backoff + jitter, BACKOFF_MAX_MS);
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = undefined;
    backoff = Math.min(backoff * 2, BACKOFF_MAX_MS);
    void connect();
  }, delay);
}

void connect();
