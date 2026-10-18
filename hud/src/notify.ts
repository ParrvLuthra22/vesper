// Native banners, posted by the HUD under its OWN app identity ("Vesper HUD", com.vesper.hud).
//
// Why: `osascript display notification` is attributed to the terminal (or "Script Editor"), which often has no notification permission, so
// a banner silently never appears. The HUD is a real app: it asks macOS for permission under its own name and is listed in System Settings >
// Notifications. The gateway sends proactive notifications to the HUD as `observation` messages carrying `native: true`, `title`, `body`;
// this module posts the banner and reports the outcome back (`hud_notify`) so the gateway knows whether to fall back to osascript.

export interface NotifyApi {
  isPermissionGranted(): Promise<boolean>;
  requestPermission(): Promise<string>;
  sendNotification(options: { title: string; body: string }): void;
}

export type NotifyState = "granted" | "denied" | "unavailable";
export type Report = (msg: { type: "hud_notify"; state: NotifyState; delivered: boolean }) => void;

/** The Tauri notification plugin, or null outside Tauri (plain-browser preview) / if the plugin is not compiled in. */
export async function loadTauriNotifyApi(): Promise<NotifyApi | null> {
  try {
    const m = await import("@tauri-apps/plugin-notification");
    return {
      isPermissionGranted: () => m.isPermissionGranted(),
      requestPermission: () => m.requestPermission() as Promise<string>,
      sendNotification: (o) => m.sendNotification(o),
    };
  } catch {
    return null;
  }
}

export class HudNotifier {
  private state: NotifyState | "unknown" = "unknown";

  constructor(
    private api: NotifyApi | null,
    private report: Report,
  ) {}

  setApi(api: NotifyApi | null): void {
    this.api = api;
  }

  /** Check (and, once, ask for) permission; tell the gateway the result. Safe to call repeatedly. */
  async ensurePermission(): Promise<NotifyState> {
    if (this.api === null) return this.set("unavailable");
    try {
      if (await this.api.isPermissionGranted()) return this.set("granted");
      const answer = await this.api.requestPermission();
      return this.set(answer === "granted" ? "granted" : "denied");
    } catch {
      return this.set("unavailable");
    }
  }

  /** Post one banner. Never throws. Reports `delivered: false` (and the state) whenever it could not even ask. */
  async deliver(title: string, body: string): Promise<boolean> {
    const state = this.state === "granted" ? "granted" : await this.ensurePermission();
    if (state !== "granted" || this.api === null) {
      this.report({ type: "hud_notify", state, delivered: false });
      return false;
    }
    try {
      this.api.sendNotification({ title, body });
      this.report({ type: "hud_notify", state: "granted", delivered: true });
      return true;
    } catch {
      this.set("unavailable");
      this.report({ type: "hud_notify", state: "unavailable", delivered: false });
      return false;
    }
  }

  /** Wire messages: act only on proactive/notify observations the gateway marked `native`. Returns true if it was one. */
  async handle(msg: any): Promise<boolean> {
    if (!msg || msg.type !== "observation" || msg.native !== true) return false;
    await this.deliver(String(msg.title ?? "Vesper"), String(msg.body ?? msg.detail ?? ""));
    return true;
  }

  private set(state: NotifyState): NotifyState {
    this.state = state;
    this.report({ type: "hud_notify", state, delivered: false });
    return state;
  }
}
