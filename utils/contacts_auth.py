"""`vesper contacts-auth`: ask macOS for Contacts access for the app this is run from, then print only the resulting status.

Why this exists: contacts_lookup checks `CNContactStore.authorizationStatusForEntityType_` FIRST and, unless the status is "authorized", returns
"not granted" WITHOUT touching the address book — so nothing ever triggered the macOS prompt and System Settings > Privacy > Contacts stayed
empty. The prompt only appears when an app calls requestAccess (or reads). This command is that explicit, user-run request. It prints no contact.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable, Optional, Sequence

STATUS_NAMES = {0: "not determined", 1: "restricted", 2: "denied", 3: "granted"}
SETTINGS = "System Settings > Privacy & Security > Contacts"


class SystemStore:
    """The real thing: the SAME Contacts framework + status call contacts_lookup uses (tools/contacts_tools.py SystemContacts)."""

    def status(self) -> Optional[int]:
        from tools.contacts_tools import SystemContacts

        return SystemContacts().authorization()

    def request(self, timeout: float = 90.0) -> Optional[bool]:
        import Contacts

        done, result = threading.Event(), {}

        def handler(granted, error):
            result["granted"] = bool(granted)
            done.set()

        Contacts.CNContactStore.alloc().init().requestAccessForEntityType_completionHandler_(Contacts.CNEntityTypeContacts, handler)
        return result.get("granted") if done.wait(timeout) else None


def run_contacts_auth(store: Any, app: str, out: Callable[[str], None] = print) -> int:
    """0 = granted. The ONLY prompt-triggering call is store.request(), and only when the status is "not determined"."""
    out(f"This will ask macOS for Contacts access for {app} (the app you are running this from).")
    before = store.status()
    if before is None:
        out("Contacts access: unknown — the Contacts framework is not available (pip install pyobjc-framework-Contacts).")
        return 3
    if before == 0:
        store.request()
    after = store.status()
    name = STATUS_NAMES.get(after, f"status {after}")
    out(f"Contacts access: {name} (held by {app}).")
    if after == 3:
        return 0
    if after == 2:
        out(f"It was denied earlier, so macOS will not ask again. Turn it on in {SETTINGS} > {app}.")
        out("If {0} is not listed, `tccutil reset Contacts` clears the stored decision — WARNING: it resets Contacts access for EVERY app, "
            "and each will ask again.".format(app))
        return 1
    if after == 1:
        out("Contacts access is restricted by a device profile; this cannot be changed from here.")
        return 1
    out("macOS did not record an answer (the prompt may have been dismissed or hidden). Run this again from a normal Terminal window.")
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    from utils.notifications import terminal_app

    return run_contacts_auth(SystemStore(), terminal_app())


if __name__ == "__main__":
    sys.exit(main())
