// @vitest-environment jsdom
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../src/gateway", () => ({
  GatewayLink: class {
    constructor() {}
    start() {}
    connect() {}
    send() {}
    onMessage() {}
    on() {}
  },
}));

const html = readFileSync(resolve(process.cwd(), "index.html"), "utf8");

describe("the HUD shows which build is running", () => {
  beforeEach(() => {
    vi.resetModules();
    const body = html.match(/<body>([\s\S]*)<\/body>/)![1].replace(/<script[\s\S]*?<\/script>/g, "");
    document.body.innerHTML = body;
  });

  it("index.html has a footer for the build id", () => {
    expect(document.getElementById("buildid")).not.toBeNull();
    expect(document.getElementById("buildid")!.tagName).toBe("FOOTER");
  });

  it("main.ts writes the injected build id into the footer and its tooltip", async () => {
    try {
      await import("../src/main");
    } catch {
      /* the rest of main.ts needs a real gateway; the build id is set first */
    }
    const el = document.getElementById("buildid")!;
    expect(el.textContent).toBe("abc1234 · 2026-10-17 23:00");
    expect(el.title).toContain("abc1234");
  });

  it("the footer is subtle and does not take pointer events", () => {
    const css = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");
    const rule = css.match(/\.buildid\s*\{([^}]*)\}/)![1];
    expect(rule).toMatch(/pointer-events:\s*none/);
    expect(rule).toMatch(/font-size:\s*[0-9.]+px/);
    expect(parseFloat(rule.match(/opacity:\s*([0-9.]+)/)![1])).toBeLessThan(0.8);
  });

  it("vite defines __BUILD_ID__ from git + time", () => {
    const cfg = readFileSync(resolve(process.cwd(), "vite.config.ts"), "utf8");
    expect(cfg).toMatch(/__BUILD_ID__/);
    expect(cfg).toMatch(/git rev-parse --short HEAD/);
  });
});
