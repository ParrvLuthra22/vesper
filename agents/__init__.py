"""
JARVIS Virtual Assistant - Agents Package.

This package contains all agent implementations:
    - BaseAgent: Abstract base class for all agents
    - VoiceAgent: Speech recognition and synthesis
    - IntentAgent: Natural language understanding
    - SystemAgent: macOS system integration
    - MemoryAgent: Context and history management

Each agent is an independent unit that communicates via the EventBus.
"""

from agents.base_agent import BaseAgent, AgentCapability, AgentState, AgentMetrics
from agents.voice_agent import VoiceAgent
from agents.intent_agent import IntentAgent
from agents.system_agent import SystemAgent
from agents.macos_control_agent import MacOSControlAgent
from agents.memory_agent import MemoryAgent
from agents.web_search_agent import WebSearchAgent
from agents.rag_agent import RAGAgent
from agents.tool_agent import ToolAgent
from agents.plugin_agent import PluginAgent

__all__ = [
    "BaseAgent",
    "AgentCapability",
    "AgentState",
    "AgentMetrics",
    "VoiceAgent",
    "IntentAgent",
    "SystemAgent",
    "MacOSControlAgent",
    "MemoryAgent",
    "WebSearchAgent",
    "RAGAgent",
    "ToolAgent",
    "PluginAgent",
]
