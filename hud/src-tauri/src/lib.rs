//! Vesper HUD — Tauri shell (PV1).
//!
//! This session builds the SHELL only: the frameless always-on-top panel, the
//! design system, the breathing star, and a live gateway link that proves data
//! flows. No feature rendering yet (that is PV2). The Rust side is deliberately
//! thin: place + reveal the panel, own the tray, and hand the frontend the
//! gateway address/token to open its own WebSocket.

use std::path::PathBuf;

use serde::Serialize;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    LogicalPosition, Manager, WebviewWindow,
};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8760;
/// Right inset from the primary display's corner, in logical px.
const INSET_X: f64 = 18.0;
/// Top inset — clears the macOS menu bar so the panel never hides behind it.
const INSET_Y: f64 = 40.0;

#[derive(Serialize, Default)]
struct GatewayConfig {
    /// ws://host:port/ws — the frontend appends ?token=… itself, since the
    /// browser WebSocket API cannot set an Authorization header.
    url: String,
    token: String,
}

#[derive(serde::Deserialize, Default)]
struct FileConfig {
    host: Option<String>,
    port: Option<u16>,
    token: Option<String>,
}

fn read_config_file(app: &tauri::AppHandle) -> Option<FileConfig> {
    // Explicit override, then the app config dir, then the current dir.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("VESPER_HUD_CONFIG") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(dir) = app.path().app_config_dir() {
        candidates.push(dir.join("config.json"));
    }
    candidates.push(PathBuf::from("vesper-hud.config.json"));

    for path in candidates {
        if let Ok(raw) = std::fs::read_to_string(&path) {
            if let Ok(cfg) = serde_json::from_str::<FileConfig>(&raw) {
                return Some(cfg);
            }
        }
    }
    None
}

/// Resolve the gateway target. Precedence: defaults < config file < environment
/// (env wins, so `VESPER_GATEWAY_TOKEN`/`_PORT` make dev launches trivial).
#[tauri::command]
fn get_gateway_config(app: tauri::AppHandle) -> GatewayConfig {
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;
    let mut token = String::new();

    if let Some(file) = read_config_file(&app) {
        if let Some(h) = file.host {
            host = h;
        }
        if let Some(p) = file.port {
            port = p;
        }
        if let Some(t) = file.token {
            token = t;
        }
    }

    if let Ok(t) = std::env::var("VESPER_GATEWAY_TOKEN") {
        if !t.is_empty() {
            token = t;
        }
    }
    if let Ok(p) = std::env::var("VESPER_GATEWAY_PORT") {
        if let Ok(parsed) = p.parse::<u16>() {
            port = parsed;
        }
    }
    if let Ok(h) = std::env::var("VESPER_GATEWAY_HOST") {
        if !h.is_empty() {
            host = h;
        }
    }

    GatewayConfig {
        url: format!("ws://{host}:{port}/ws"),
        token,
    }
}

fn toggle_visibility(window: &WebviewWindow) {
    if window.is_visible().unwrap_or(false) {
        let _ = window.hide();
    } else {
        let _ = window.show();
    }
}

#[tauri::command]
fn toggle_panel(window: WebviewWindow) {
    toggle_visibility(&window);
}

#[tauri::command]
fn quit_app(app: tauri::AppHandle) {
    app.exit(0);
}

/// Make the panel a true HUD: float over every Space, including other apps'
/// fullscreen windows. Tauri's set_visible_on_all_workspaces only sets
/// canJoinAllSpaces; a HUD also needs fullScreenAuxiliary.
#[cfg(target_os = "macos")]
fn float_over_fullscreen(window: &WebviewWindow) {
    use objc2::msg_send;
    use objc2::runtime::AnyObject;

    // NSWindowCollectionBehavior bit flags:
    //   CanJoinAllSpaces (1<<0) | Stationary (1<<4) | FullScreenAuxiliary (1<<8)
    const CAN_JOIN_ALL_SPACES: usize = 1 << 0;
    const STATIONARY: usize = 1 << 4;
    const FULLSCREEN_AUXILIARY: usize = 1 << 8;
    let behavior: usize = CAN_JOIN_ALL_SPACES | STATIONARY | FULLSCREEN_AUXILIARY;

    if let Ok(ptr) = window.ns_window() {
        let ns_window = ptr as *mut AnyObject;
        if !ns_window.is_null() {
            unsafe {
                let _: () = msg_send![ns_window, setCollectionBehavior: behavior];
            }
        }
    }
}

/// Pin the panel to the top-right of the primary display with a small inset.
fn position_top_right(window: &WebviewWindow) {
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let scale = monitor.scale_factor();
        let screen = monitor.size().to_logical::<f64>(scale);
        if let Ok(size) = window.outer_size() {
            let win = size.to_logical::<f64>(scale);
            let x = (screen.width - win.width - INSET_X).max(0.0);
            let _ = window.set_position(LogicalPosition::new(x, INSET_Y));
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            get_gateway_config,
            toggle_panel,
            quit_app
        ])
        .setup(|app| {
            let window = app
                .get_webview_window("main")
                .expect("main window must exist");
            position_top_right(&window);
            // A HUD should float across every Space — including over other
            // apps' fullscreen windows — not just the active desktop.
            let _ = window.set_visible_on_all_workspaces(true);
            #[cfg(target_os = "macos")]
            float_over_fullscreen(&window);
            let _ = window.show();

            // Tray: show/hide the panel, and quit. Left-click opens the menu.
            let toggle_item =
                MenuItem::with_id(app, "toggle", "Show / Hide Panel", true, None::<&str>)?;
            let quit_item =
                MenuItem::with_id(app, "quit", "Quit Vesper HUD", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&toggle_item, &quit_item])?;

            let _tray = TrayIconBuilder::with_id("vesper-hud-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .tooltip("Vesper")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "toggle" => {
                        if let Some(window) = app.get_webview_window("main") {
                            toggle_visibility(&window);
                        }
                    }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
