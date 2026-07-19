"""
VESPER Virtual Assistant - Orchestrator Package.

This package contains the Brain orchestrator that coordinates all agents
and the Planner (LLM tool-calling loop) that decides what to do with user
text.
"""

__all__ = [
    "Brain",
    "BrainState",
    "create_brain",
]


def __getattr__(name: str):
    if name in {"Brain", "BrainState", "create_brain"}:
        from orchestrator.brain import Brain, BrainState, create_brain

        mapping = {
            "Brain": Brain,
            "BrainState": BrainState,
            "create_brain": create_brain,
        }
        return mapping[name]
    raise AttributeError(f"module 'orchestrator' has no attribute '{name}'")
