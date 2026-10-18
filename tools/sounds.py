"""play_sound — play one of macOS's own alert sounds. Safe tier: it can only run `afplay` on a file from an allow-list under
/System/Library/Sounds. No arbitrary paths, no shell (argv list). The result says the command ran ("played Tink (exit 0)"); it NEVER
claims the sound was heard — the volume may be down or the output device elsewhere (the claim verifier flags "you should hear it")."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from tools.registry import ToolSpec, get_registry
from utils.grounding import grounded

SOUND_DIR = Path("/System/Library/Sounds")
#: The only names that can be played (the files macOS ships there).
ALLOWED = ("Basso", "Blow", "Bottle", "Frog", "Funk", "Glass", "Hero", "Morse", "Ping", "Pop", "Purr", "Sosumi", "Submarine", "Tink")
#: Spoken words -> sound.
ALIASES = {"beep": "Tink", "ping": "Ping", "ding": "Glass", "chime": "Glass", "pop": "Pop", "boop": "Pop", "buzz": "Basso",
           "sound": "Tink", "noise": "Tink", "alert": "Sosumi", "default": "Tink"}


def resolve(name: object) -> Optional[str]:
    """A canonical allow-listed sound name, or None. Case-insensitive; accepts the spoken aliases; refuses anything else."""
    key = str(name or "").strip().lower()
    for allowed in ALLOWED:
        if key == allowed.lower():
            return allowed
    return ALIASES.get(key)


def make_handler(run: Optional[Callable[[Sequence], Tuple[bool, str]]] = None, sound_dir: Path = SOUND_DIR):
    def _afplay(argv):
        from utils import safe_exec

        ok, _out, err = safe_exec.run_argv(list(argv), timeout=10)
        return ok, err

    runner = run or _afplay

    async def handler(arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        wanted = (arguments or {}).get("name") or "beep"
        name = resolve(wanted)
        if name is None:
            return f"I only play these system sounds: {', '.join(ALLOWED)}. That name is not one of them."
        path = sound_dir / f"{name}.aiff"
        if not path.exists():
            return f"The {name} sound file is not on this Mac."
        ok, err = await asyncio.get_running_loop().run_in_executor(None, runner, ["afplay", str(path)])
        if not ok:
            return grounded(f"Could not play {name} ({(err or 'afplay failed')[:60]})", can=[], cannot=["sound_audible"])
        return grounded(f"played {name} (exit 0)", can=["sound_played"], cannot=["sound_audible"])
    return handler


def _register() -> None:
    if get_registry().get("play_sound") is None:
        get_registry().register(ToolSpec(
            name="play_sound",
            description=("Play a short macOS system sound (a beep/ping) through the speakers. Names: " + ", ".join(ALLOWED) + ". "
                         "Reports that the player ran, never that it was heard."),
            parameters={"type": "object", "properties": {"name": {"type": "string", "enum": list(ALLOWED),
                                                                  "description": "Which sound; default Tink."}}},
            tier="safe", handler=make_handler(), category="system",
        ))


_register()
