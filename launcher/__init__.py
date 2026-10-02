"""Vesper launcher: one command to start, supervise and stop the whole stack.

    vesper up        # gateway -> voice output -> HUD -> voice input (wake word + STT)
    vesper status    # what is running, restarts, uptime
    vesper down      # clean, ordered shutdown

See launcher/supervisor.py for the process model and docs/ARCHITECTURE.md.
"""
