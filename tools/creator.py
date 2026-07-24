"""Vesper's creator tools (PC2) — deep research, script writer, and a guarded
automation composer. Registered into the shared registry on import (see
tools/__init__.py). Native handlers; no planner/gateway/HUD/voice changes.

  - research(question, depth)  — MMAS-style Search→Reader→Writer→Critic pipeline,
    slow=True (task queue → "…is ready" observation), full report to a file.
  - write_script(topic, format) — template-driven from config/formats/<format>.md.
  - run_shell(command), run_applescript(script) — tier DANGEROUS: the Guardian
    shows the FULL command/script verbatim, and a safety denylist refuses
    dangerous patterns OUTRIGHT — even after approval.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import ToolSpec, get_registry
from utils.logger import get_logger

logger = get_logger(__name__)
registry = get_registry()


# ------------------------------- shared bits -------------------------------
def _cfg(key: str, default: Any = None) -> Any:
    from config.settings import load_config_dict

    value: Any = load_config_dict()
    for part in key.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
        if value is None:
            return default
    return value


def get_router():
    """ModelRouter from config. Split out so tests can monkeypatch it."""
    from config.settings import load_config_dict
    from llm.router import ModelRouter

    return ModelRouter(config=load_config_dict())


async def _llm(prompt: str, purpose: str = "planning") -> str:
    resp = await get_router().complete(messages=[{"role": "user", "content": prompt}], purpose=purpose)
    return (getattr(resp, "text", "") or "").strip()


async def _run(cmd: List[str], timeout: float = 120.0) -> Tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"timed out after {timeout:.0f}s"
    return proc.returncode or 0, out.decode(errors="replace"), err.decode(errors="replace")


def _slug(text: str, n: int = 48) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n] or "untitled").strip("-")


def _store(dir_key_default: str, title: str, body: str, ext: str = "md") -> str:
    directory = Path(dir_key_default)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}-{_slug(title)}.{ext}"
    path.write_text(body, encoding="utf-8")
    return str(path)


# ================================ 1. RESEARCH ==============================
# Web search (Tavily) + fetch. Both are module-level so tests can mock them.
def _tavily_search(query: str, max_results: int) -> List[Dict[str, str]]:
    key = os.getenv("TAVILY_API_KEY") or _cfg("web_search.tavily_api_key")
    if not key:
        return []
    try:
        from tavily import TavilyClient

        res = TavilyClient(api_key=key).search(query, max_results=max_results, include_answer=False)
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in res.get("results", [])
        ]
    except Exception as exc:
        logger.warning(f"[research] tavily search failed: {exc}")
        return []


async def _search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    return await asyncio.get_event_loop().run_in_executor(None, _tavily_search, query, max_results)


async def _fetch(url: str) -> str:
    import urllib.request

    def _get() -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Vesper/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read(400_000).decode(errors="replace")
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    try:
        return await asyncio.get_event_loop().run_in_executor(None, _get)
    except Exception as exc:
        logger.warning(f"[research] fetch failed for {url}: {exc}")
        return ""


async def _plan_subqueries(question: str, n: int) -> List[str]:
    if n <= 1:
        return [question]
    text = await _llm(
        f"Break this research question into {n} focused, distinct web-search queries. "
        f"One per line, no numbering.\n\nQuestion: {question}"
    )
    qs = [ln.strip("-•* ").strip() for ln in text.splitlines() if ln.strip()][:n]
    return qs or [question]


async def _read_sources(sources: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
    read: List[Dict[str, str]] = []
    for s in sources[:k]:
        body = (await _fetch(s.get("url", ""))) or s.get("content", "")
        if body:
            read.append({"title": s.get("title", ""), "url": s.get("url", ""), "text": body[:3000]})
    return read


async def _writer(question: str, read: List[Dict[str, str]]) -> str:
    src = "\n\n".join(f"[{i + 1}] {r['title']} ({r['url']})\n{r['text']}" for i, r in enumerate(read))
    return await _llm(
        "Write a thorough, well-organized answer to the question using ONLY the sources below. "
        "Cite every non-obvious claim with [n]. End with a 'Sources' section listing each [n] as "
        f"'title — URL'.\n\nQuestion: {question}\n\nSources:\n{src}"
    )


async def _critic(question: str, draft: str) -> Tuple[str, str]:
    text = await _llm(
        "You are a critical editor. Review this research answer for unsupported claims, missing "
        "citations, and gaps; produce a FINAL corrected version. Then on the very last lines write "
        "'=== SUMMARY ===' followed by EXACTLY three short lines summarizing the answer for a spoken "
        f"briefing.\n\nQuestion: {question}\n\nDraft:\n{draft}"
    )
    if "=== SUMMARY ===" in text:
        final, summ = text.split("=== SUMMARY ===", 1)
        return final.strip(), "\n".join(summ.strip().splitlines()[:3]).strip()
    return text.strip(), "\n".join(text.strip().splitlines()[:3]).strip()


async def research(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    question = str(arguments.get("question", "")).strip()
    if not question:
        return "No research question given, Sir."
    deep = str(arguments.get("depth", "standard")).lower() == "deep"
    n_queries, k_read = (4, 8) if deep else (1, 4)

    sources: List[Dict[str, str]] = []
    seen = set()
    for q in await _plan_subqueries(question, n_queries):
        for s in await _search(q, max_results=5):
            u = s.get("url", "")
            if u and u not in seen:
                seen.add(u)
                sources.append(s)
    if not sources:
        return "I could not retrieve any sources, Sir — web search may be unavailable (no TAVILY_API_KEY)."

    read = await _read_sources(sources, k_read)
    draft = await _writer(question, read)
    final, summary3 = await _critic(question, draft)

    report = f"# Research: {question}\n\n_Depth: {'deep' if deep else 'standard'} · {len(read)} sources read_\n\n{final}\n"
    path = _store(_cfg("creator.research_dir", "data/research"), question, report)
    return f"{summary3}\n\nFull report: {path}"


# ============================== 2. SCRIPT WRITER ============================
async def write_script(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    topic = str(arguments.get("topic", "")).strip()
    fmt = (str(arguments.get("format", "reel")).strip() or "reel")
    if not topic:
        return "No topic given, Sir."
    spec_path = Path(_cfg("creator.formats_dir", "config/formats")) / f"{fmt}.md"
    if not spec_path.exists():
        return f"No format spec for '{fmt}', Sir — add {spec_path} (formats are just markdown files)."

    spec = spec_path.read_text(encoding="utf-8")
    script = await _llm(
        "You are writing a short-form script. Obey this format specification EXACTLY — the beat "
        "order, every identity rule, the signature line, and the sign-off. Do not add anything the "
        f"spec forbids.\n\n=== FORMAT SPEC ===\n{spec}\n=== END SPEC ===\n\nTopic: {topic}\n\nWrite the script now."
    )
    path = _store(_cfg("creator.scripts_dir", "data/scripts"), f"{fmt}-{topic}", script)
    preview = "\n".join([ln for ln in script.splitlines() if ln.strip()][:6])
    return f"{preview}\n…\nFull script: {path}"


# ========================== 3. AUTOMATION COMPOSER =========================
# A denylist of patterns refused OUTRIGHT — regardless of confirmation. This is
# a safety floor (always enforced); config `creator.automation.denylist` only
# ADDS more patterns, it can never weaken these.
DEFAULT_DENYLIST: List[str] = [
    r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?[a-zA-Z]*\s+(/|~|/\*|\$HOME)",   # rm -rf /  /  rm -rf ~
    r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*\s+(/|~|/\*|\$HOME)",     # rm -fr /
    r"\bsudo\b",
    r"\b(diskutil|asr|fdisk|gpt|newfs\w*|mkfs\w*)\b",               # disk utilities
    r"\bdd\b[^\n]*\bof=/dev/",                                       # dd of=/dev/...
    r"(curl|wget)\s[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?)\b",   # curl … | sh
    r">\s*/dev/r?disk",
    r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}",                       # fork bomb
]

_SAFE_WRITE_PREFIXES = ("/tmp", "/private/tmp", "/var/folders")


def _writes_outside_home(cmd: str) -> Optional[str]:
    home = os.path.expanduser("~")
    prefixes = (home,) + _SAFE_WRITE_PREFIXES

    def outside(p: str) -> bool:
        return p.startswith("/") and not any(p.startswith(sp) for sp in prefixes)

    for m in re.finditer(r">>?\s*(/[^\s;|&>]+)", cmd):  # redirects to an absolute path
        if outside(m.group(1)):
            return f"redirect to {m.group(1)}"
    for m in re.finditer(r"\b(rm|mv|cp|tee|touch|mkdir|rmdir|chmod|chown|ln)\b([^\n;|&]*)", cmd):
        for tok in re.findall(r"/[^\s]+", m.group(2)):
            if outside(tok):
                return f"{m.group(1)} -> {tok}"
    return None


def _denylist_match(text: str) -> Optional[str]:
    patterns = DEFAULT_DENYLIST + [str(p) for p in (_cfg("creator.automation.denylist", []) or [])]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    if _cfg("creator.automation.enforce_home_boundary", True):
        outside = _writes_outside_home(text)
        if outside:
            return f"writes outside home ({outside})"
    return None


def _verbatim_summary(kind: str, body: str) -> str:
    blocked = _denylist_match(body)
    if blocked:
        return f"⛔ REFUSED by the safety denylist ({blocked}) — this will NOT run even if approved:\n\n{body}"
    return f"Run this {kind} verbatim (DANGEROUS — approve every time):\n\n{body}"


async def _shell_summary(arguments: Dict[str, Any]) -> str:
    return _verbatim_summary("shell command", str(arguments.get("command", "")))


async def _applescript_summary(arguments: Dict[str, Any]) -> str:
    return _verbatim_summary("AppleScript", str(arguments.get("script", "")))


async def run_shell(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    cmd = str(arguments.get("command", "")).strip()
    if not cmd:
        return "No command given."
    blocked = _denylist_match(cmd)
    if blocked:
        return f"Refused — the command matches the safety denylist ({blocked}) and was NOT executed."
    rc, out, err = await _run(["sh", "-c", cmd], timeout=120)
    body = (out.strip() or err.strip() or "(no output)")[:2000]
    return f"$ {cmd}\n{body}" if rc == 0 else f"$ {cmd}\nexit {rc}\n{(err or out).strip()[:2000]}"


async def run_applescript(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
    script = str(arguments.get("script", "")).strip()
    if not script:
        return "No script given."
    blocked = _denylist_match(script)
    if blocked:
        return f"Refused — the script matches the safety denylist ({blocked}) and was NOT executed."
    rc, out, err = await _run(["osascript", "-e", script], timeout=120)
    return (out.strip() or "AppleScript completed.") if rc == 0 else f"AppleScript failed: {(err or out).strip()[:400]}"


# ============================== registration ==============================
def _register() -> None:
    registry.register(ToolSpec(
        name="research",
        description=(
            "Deep multi-step web research (Search → Reader → Writer → Critic). Returns a short "
            "sourced summary and writes the full report to a file. depth='deep' does broader "
            "searching. Runs in the background; the user is told when it is ready."
        ),
        parameters={"type": "object", "properties": {
            "question": {"type": "string", "description": "The research question."},
            "depth": {"type": "string", "enum": ["standard", "deep"], "description": "Research depth."},
        }, "required": ["question"]},
        tier="safe", handler=research, category="creator", slow=True,
    ))
    registry.register(ToolSpec(
        name="write_script",
        description=(
            "Write a short-form script from a template. `format` selects the spec file in "
            "config/formats/<format>.md (e.g. 'reel'). Obeys that spec exactly. Saves the full "
            "script to a file and returns a short preview."
        ),
        parameters={"type": "object", "properties": {
            "topic": {"type": "string", "description": "What the script is about."},
            "format": {"type": "string", "description": "Format spec name (default 'reel')."},
        }, "required": ["topic"]},
        tier="safe", handler=write_script, category="creator",
    ))
    registry.register(ToolSpec(
        name="run_shell",
        description=(
            "Execute a shell command composed for the user. DANGEROUS: requires explicit approval "
            "every time (the exact command is shown), and refuses denylisted commands outright."
        ),
        parameters={"type": "object", "properties": {
            "command": {"type": "string", "description": "The exact shell command to run."}}, "required": ["command"]},
        tier="dangerous", handler=run_shell, confirm_summary=_shell_summary, category="creator",
    ))
    registry.register(ToolSpec(
        name="run_applescript",
        description=(
            "Execute an AppleScript composed for the user. DANGEROUS: requires explicit approval "
            "every time (the exact script is shown), and refuses denylisted scripts outright."
        ),
        parameters={"type": "object", "properties": {
            "script": {"type": "string", "description": "The exact AppleScript to run."}}, "required": ["script"]},
        tier="dangerous", handler=run_applescript, confirm_summary=_applescript_summary, category="creator",
    ))


_register()
