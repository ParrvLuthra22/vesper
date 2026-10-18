import { describe, expect, it } from "vitest";
import { HudNotifier, type NotifyApi } from "../src/notify";

function fakeApi(opts: { granted?: boolean; ask?: string; throwOnSend?: boolean } = {}) {
  const sent: Array<{ title: string; body: string }> = [];
  const calls = { ask: 0 };
  const api: NotifyApi = {
    isPermissionGranted: async () => opts.granted ?? false,
    requestPermission: async () => {
      calls.ask++;
      return opts.ask ?? "denied";
    },
    sendNotification: (o) => {
      if (opts.throwOnSend) throw new Error("boom");
      sent.push(o);
    },
  };
  return { api, sent, calls };
}

function make(api: NotifyApi | null) {
  const reports: any[] = [];
  return { n: new HudNotifier(api, (m) => reports.push(m)), reports };
}

describe("HudNotifier (the HUD posts banners under its own app identity)", () => {
  it("HUD connected + permission granted: posts the banner and reports delivered", async () => {
    const { api, sent } = fakeApi({ granted: true });
    const { n, reports } = make(api);
    expect(await n.deliver("Vesper · Slack", "New message")).toBe(true);
    expect(sent).toEqual([{ title: "Vesper · Slack", body: "New message" }]);
    expect(reports.at(-1)).toEqual({ type: "hud_notify", state: "granted", delivered: true });
  });

  it("asks for permission once, then posts when macOS grants it", async () => {
    const { api, sent, calls } = fakeApi({ granted: false, ask: "granted" });
    const { n } = make(api);
    await n.deliver("a", "b");
    await n.deliver("c", "d");
    expect(calls.ask).toBe(1);
    expect(sent.length).toBe(2);
  });

  it("denied: nothing is posted and the gateway is told to fall back", async () => {
    const { api, sent } = fakeApi({ granted: false, ask: "denied" });
    const { n, reports } = make(api);
    expect(await n.deliver("t", "b")).toBe(false);
    expect(sent).toEqual([]);
    expect(reports.at(-1)).toEqual({ type: "hud_notify", state: "denied", delivered: false });
  });

  it("plugin missing (plain browser / not compiled in): unavailable, never throws", async () => {
    const { n, reports } = make(null);
    expect(await n.deliver("t", "b")).toBe(false);
    expect(reports.at(-1)).toEqual({ type: "hud_notify", state: "unavailable", delivered: false });
  });

  it("a plugin that throws on send is reported unavailable (fallback)", async () => {
    const { api } = fakeApi({ granted: true, throwOnSend: true });
    const { n, reports } = make(api);
    expect(await n.deliver("t", "b")).toBe(false);
    expect(reports.at(-1)).toEqual({ type: "hud_notify", state: "unavailable", delivered: false });
  });

  it("handles only native observations; ordinary messages are ignored", async () => {
    const { api, sent } = fakeApi({ granted: true });
    const { n } = make(api);
    expect(await n.handle({ type: "reply", text: "hello" })).toBe(false);
    expect(await n.handle({ type: "observation", kind: "context_switch", detail: "x" })).toBe(false);
    expect(await n.handle({ type: "observation", native: true, title: "Vesper", body: "Tea is ready", detail: "Tea is ready" })).toBe(true);
    expect(sent).toEqual([{ title: "Vesper", body: "Tea is ready" }]);
  });

  it("permission check on connect reports the state without posting anything", async () => {
    const { api, sent } = fakeApi({ granted: true });
    const { n, reports } = make(api);
    expect(await n.ensurePermission()).toBe("granted");
    expect(sent).toEqual([]);
    expect(reports[0]).toEqual({ type: "hud_notify", state: "granted", delivered: false });
  });
});
