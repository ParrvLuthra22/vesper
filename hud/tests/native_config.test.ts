import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const read = (p: string) => readFileSync(resolve(process.cwd(), p), "utf8");

describe("the Tauri shell is configured for native banners", () => {
  it("compiles the notification plugin in and initialises it", () => {
    expect(read("src-tauri/Cargo.toml")).toMatch(/tauri-plugin-notification\s*=\s*"2"/);
    expect(read("src-tauri/src/lib.rs")).toMatch(/\.plugin\(tauri_plugin_notification::init\(\)\)/);
  });
  it("grants the main window the notification permission", () => {
    const cap = JSON.parse(read("src-tauri/capabilities/default.json"));
    expect(cap.permissions).toContain("notification:default");
    expect(cap.permissions).toContain("core:default");
  });
  it("depends on the JS plugin", () => {
    expect(JSON.parse(read("package.json")).dependencies["@tauri-apps/plugin-notification"]).toMatch(/^\^2/);
  });
  it("the HUD announces itself so the gateway can route banners here", () => {
    expect(read("src/gateway.ts")).toMatch(/type: "hello", client: "hud"/);
    expect(read("src/main.ts")).toMatch(/notifier\.handle\(msg\)/);
  });
});
