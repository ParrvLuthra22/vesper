// Gateway link — config resolution, resilient WebSocket, and the REST
// /confirm POST. The Brain is the source of truth; this is a stateless view.

import { invoke } from "@tauri-apps/api/core";

export type ConnState = "connecting" | "connected" | "disconnected";

export interface GatewayTarget {
  wsUrl: string; // ws://host:port/ws?token=…
  httpBase: string; // http://host:port
  token: string;
  port: string;
}

export async function resolveGateway(): Promise<GatewayTarget> {
  let url = "ws://127.0.0.1:8760/ws";
  let token = "";
  try {
    const cfg = await invoke<{ url: string; token: string }>("get_gateway_config");
    url = cfg.url || url;
    token = cfg.token || "";
  } catch {
    // not in Tauri (plain-browser preview) — defaults; socket will ember out.
  }
  const port = url.match(/:(\d+)\//)?.[1] ?? "";
  const wsUrl = token ? `${url}?token=${encodeURIComponent(token)}` : url;
  const httpBase = url.replace(/^ws/, "http").replace(/\/ws$/, "");
  return { wsUrl, httpBase, token, port };
}

type WireHandler = (msg: any) => void;
type ConnHandler = (state: ConnState, port: string) => void;

const BACKOFF_MIN = 500;
const BACKOFF_MAX = 10_000;

export class GatewayLink {
  private target: GatewayTarget | null = null;
  private ws: WebSocket | null = null;
  private backoff = BACKOFF_MIN;
  private timer: number | undefined;

  constructor(
    private onWire: WireHandler,
    private onConn: ConnHandler,
  ) {}

  async start(): Promise<void> {
    await this.connect();
  }

  private async connect(): Promise<void> {
    if (!this.target) this.target = await resolveGateway();
    const target = this.target;
    this.onConn("connecting", target.port);

    let ws: WebSocket;
    try {
      ws = new WebSocket(target.wsUrl);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.onopen = () => {
      this.backoff = BACKOFF_MIN;
      this.onConn("connected", target.port);
    };
    ws.onmessage = (ev) => {
      try {
        this.onWire(JSON.parse(ev.data as string));
      } catch {
        /* ignore non-JSON */
      }
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch {
        /* noop */
      }
    };
    ws.onclose = () => {
      this.onConn("disconnected", target.port);
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    if (this.timer !== undefined) return;
    const jitter = Math.random() * 0.3 * this.backoff;
    const delay = Math.min(this.backoff + jitter, BACKOFF_MAX);
    this.timer = window.setTimeout(() => {
      this.timer = undefined;
      this.backoff = Math.min(this.backoff * 2, BACKOFF_MAX);
      void this.connect();
    }, delay);
  }

  /** Send a short command as a user turn. Returns false if not connected. */
  sendMessage(text: string): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "message", text }));
      return true;
    }
    return false;
  }

  /** Approve/deny a Guardian confirmation via REST POST /confirm. */
  async confirm(requestId: string, approved: boolean): Promise<boolean> {
    if (!this.target) return false;
    try {
      const res = await fetch(`${this.target.httpBase}/confirm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.target.token ? { Authorization: `Bearer ${this.target.token}` } : {}),
        },
        body: JSON.stringify({ request_id: requestId, approved }),
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}
