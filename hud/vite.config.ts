import { defineConfig } from "vite";
import { execSync } from "node:child_process";

// Build id shown in the HUD footer, so a screenshot proves which version is running: git short hash (+ "+" when the working tree
// has uncommitted changes) and the build time.
function buildId(): string {
  let hash = "nogit";
  let dirty = "";
  try {
    hash = execSync("git rev-parse --short HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    dirty = execSync("git status --porcelain -- .", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim() ? "+" : "";
  } catch {
    /* not a git checkout */
  }
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${hash}${dirty} · ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({
  define: { __BUILD_ID__: JSON.stringify(buildId()) },

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },
}));
