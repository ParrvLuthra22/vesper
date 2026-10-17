import { defineConfig } from "vitest/config";

export default defineConfig({ define: { __BUILD_ID__: JSON.stringify("abc1234 · 2026-10-17 23:00") }, test: { environment: "jsdom", include: ["tests/**/*.test.ts"] } });
