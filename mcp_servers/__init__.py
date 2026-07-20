"""
MCP servers — each one a standalone subprocess (its own Python 3.10+
virtualenv at mcp_servers/.venv, since the official `mcp` SDK requires
3.10+ and the main app runs on 3.9). tools/mcp_bridge.py in the main app
talks to these over stdio using the plain MCP wire protocol; nothing in
the main app imports these modules directly.
"""
