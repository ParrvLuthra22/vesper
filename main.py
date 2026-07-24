#!/usr/bin/env python3
"""
VESPER Virtual Assistant - Main Entry Point.

This is the main entry point for the VESPER virtual assistant.
It initializes the system, loads configuration, and starts the Brain
which orchestrates all agents.

Usage:
    python main.py
    python main.py --config /path/to/custom/settings.yaml
    python main.py --debug

Features:
    - Configuration loading from YAML
    - Environment variable overrides
    - Graceful shutdown handling with signal trapping
    - Centralized error handling with recovery
    - Debug mode for development
    - Stable infinite loop with watchdog

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                         Brain                                │
    │                    (Orchestrator)                            │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │                    Event Bus                             │ │
    │  │              (Pub/Sub Communication)                     │ │
    │  └─────────────────────────────────────────────────────────┘ │
    │         ▲              ▲              ▲              ▲       │
    │         │              │              │              │       │
    │    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐   │
    │    │  Voice  │   │ Intent  │   │ System  │   │ Memory  │   │
    │    │  Agent  │   │  Agent  │   │  Agent  │   │  Agent  │   │
    │    └─────────┘   └─────────┘   └─────────┘   └─────────┘   │
    └─────────────────────────────────────────────────────────────┘
    
    Flow:
    1. Voice Agent captures speech → VoiceInputEvent
    2. Intent Agent classifies → IntentRecognizedEvent
    3. System Agent executes → SystemCommandResultEvent
    4. Voice Agent speaks result ← VoiceOutputEvent
    5. Memory Agent tracks context throughout

Lifecycle:
    1. Load configuration from YAML
    2. Initialize EventBus singleton
    3. Create Brain orchestrator
    4. Brain registers default agents (Memory, System, Intent, Voice)
    5. Brain starts agents in startup order
    6. Main loop runs indefinitely with health monitoring
    7. On SIGINT/SIGTERM: graceful shutdown in reverse order
"""

from __future__ import annotations

import asyncio
import atexit
import multiprocessing
import os
import platform
import signal
import sys
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set multiprocessing start method for macOS (required for OpenCV GUI in subprocess)
if platform.system() == "Darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed, skip

# Global shutdown event for coordinated shutdown
_shutdown_event: Optional[asyncio.Event] = None
_brain_instance: Optional[Any] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Priority:
    1. Specified config path
    2. VESPER_CONFIG environment variable
    3. Default: config/settings.yaml
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Configuration dictionary
    
    TODO: Add environment variable overrides
    TODO: Add config schema validation
    """
    from config.settings import load_config_dict

    # Determine config path
    if config_path is None:
        config_path = os.environ.get(
            "VESPER_CONFIG",
            str(PROJECT_ROOT / "config" / "settings.yaml")
        )

    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        return load_config_dict(None)

    try:
        config = load_config_dict(config_path)
        print(f"Loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return load_config_dict(None)


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config: Configuration dictionary
    """
    from utils.logger import init_from_config
    
    general_config = config.get("general", {})
    
    # Check for debug mode
    debug_mode = (
        general_config.get("debug_mode", False)
        or os.environ.get("VESPER_DEBUG", "").lower() in ("1", "true", "yes")
        or "--debug" in sys.argv
    )
    
    if debug_mode:
        general_config["log_level"] = "DEBUG"
        print("Debug mode enabled")
    
    init_from_config(general_config)


def log_startup_preflight(config: Dict[str, Any], logger: Any) -> None:
    """
    Log critical startup feature switches and key availability.

    This helps confirm .env keys are being applied when launching with
    `python3 main.py`.
    """
    from utils.api_keys import get_gemini_api_key, get_openrouter_api_key

    wake_word = str(config.get("voice", {}).get("wake_word", "vesper"))
    tavily_key = (
        os.getenv("TAVILY_API_KEY")
        or config.get("web_search", {}).get("tavily_api_key")
        or config.get("system", {}).get("apis", {}).get("tavily", {}).get("api_key")
    )
    openrouter_key = get_openrouter_api_key(lambda key: _dot_get(config, key))
    gemini_key = get_gemini_api_key(lambda key: _dot_get(config, key))

    logger.info(f"Startup preflight | wake_word={wake_word}")
    logger.info(
        "Startup preflight keys | "
        f"TAVILY_API_KEY={'set' if bool(tavily_key) else 'missing'} | "
        f"OPENROUTER_API_KEY={'set' if bool(openrouter_key) else 'missing'} | "
        f"GEMINI_API_KEY={'set' if bool(gemini_key) else 'missing'}"
    )


def _dot_get(config: Dict[str, Any], key: str) -> Any:
    """Read nested dict values via dot notation."""
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def print_banner() -> None:
    """Print startup banner."""
    banner = """
    ╔═════════════════════════════════════════════════════════════════╗
    ║                                                                 ║
    ║        ██╗   ██╗███████╗███████╗██████╗ ███████╗██████╗         ║
    ║        ██║   ██║██╔════╝██╔════╝██╔══██╗██╔════╝██╔══██╗        ║
    ║        ██║   ██║█████╗  ███████╗██████╔╝█████╗  ██████╔╝        ║
    ║        ╚██╗ ██╔╝██╔══╝  ╚════██║██╔═══╝ ██╔══╝  ██╔══██╗        ║
    ║         ╚████╔╝ ███████╗███████║██║     ███████╗██║  ██║        ║
    ║          ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝        ║
    ║                                                                 ║
    ║                   Virtual Assistant for macOS                   ║
    ║                                                                 ║
    ╚═════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    
    Returns:
        True if all required dependencies are available
    
    TODO: Add more comprehensive dependency checking
    """
    missing = []
    
    # Check for PyYAML
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    # Check for optional but recommended packages
    optional_missing = []
    
    try:
        from google import genai
    except ImportError:
        optional_missing.append("google-genai (for Gemini-based intent recognition)")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    if optional_missing:
        print("\nOptional dependencies not installed:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print()
    
    return True


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments.
    
    Returns:
        Dictionary of parsed arguments
    """
    args = {
        "config": None,
        "debug": False,
    }
    
    argv = sys.argv[1:]
    i = 0
    
    while i < len(argv):
        arg = argv[i]
        
        if arg in ("--config", "-c") and i + 1 < len(argv):
            args["config"] = argv[i + 1]
            i += 2
        elif arg in ("--debug", "-d"):
            args["debug"] = True
            i += 1
        elif arg in ("--help", "-h"):
            print_help()
            sys.exit(0)
        else:
            i += 1
    
    return args


def print_help() -> None:
    """Print help message."""
    help_text = """
VESPER Virtual Assistant

Usage:
    python main.py [options]

Options:
    -c, --config PATH   Path to configuration file
    -d, --debug         Enable debug mode
    -h, --help          Show this help message

Environment Variables:
    VESPER_CONFIG       Path to configuration file
    VESPER_DEBUG        Enable debug mode (1/true/yes)
    GEMINI_API_KEY      Gemini API key for intent recognition

Examples:
    python main.py
    python main.py --debug
    python main.py --config /path/to/settings.yaml
"""
    print(help_text)


# =============================================================================
# Graceful Shutdown Handling
# =============================================================================

def _create_shutdown_handler(loop: asyncio.AbstractEventLoop, logger: Any) -> None:
    """
    Set up signal handlers for graceful shutdown.
    
    Handles SIGINT (Ctrl+C) and SIGTERM for clean termination.
    """
    global _shutdown_event
    
    def signal_handler(sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        sig_name = sig.name if hasattr(sig, 'name') else str(sig)
        logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
        
        if _shutdown_event and not _shutdown_event.is_set():
            _shutdown_event.set()
    
    # Register handlers for both SIGINT and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler(s))


async def _graceful_shutdown(brain: Any, logger: Any, reason: str = "User interrupt") -> None:
    """
    Perform graceful shutdown of the Brain and all agents.
    
    Args:
        brain: The Brain instance to shut down
        logger: Logger instance
        reason: Reason for shutdown
    """
    logger.info(f"Graceful shutdown initiated: {reason}")
    
    try:
        # Give agents time to complete in-flight operations
        await asyncio.wait_for(brain.stop(reason), timeout=10.0)
        logger.info("All agents stopped successfully")
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out, forcing termination")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# =============================================================================
# Error Recovery
# =============================================================================

class ErrorRecovery:
    """
    Centralized error handling and recovery.
    
    Tracks errors and decides whether to:
    - Retry operations
    - Restart specific agents
    - Perform full system restart
    - Give up and exit
    """
    
    MAX_CONSECUTIVE_ERRORS = 5
    ERROR_WINDOW_SECONDS = 60
    
    def __init__(self, logger: Any):
        self.logger = logger
        self._error_timestamps: list[datetime] = []
        self._consecutive_errors = 0
    
    def record_error(self, error: Exception, context: str = "") -> bool:
        """
        Record an error and decide if system should continue.
        
        Args:
            error: The exception that occurred
            context: Description of where error occurred
            
        Returns:
            True if system should continue, False if should exit
        """
        now = datetime.now()
        self._error_timestamps.append(now)
        self._consecutive_errors += 1
        
        # Clean old errors outside the window
        cutoff = now.timestamp() - self.ERROR_WINDOW_SECONDS
        self._error_timestamps = [
            t for t in self._error_timestamps 
            if t.timestamp() > cutoff
        ]
        
        # Log the error
        self.logger.error(
            f"Error in {context}: {error}\n{traceback.format_exc()}"
        )
        
        # Decide if we should continue
        if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            self.logger.critical(
                f"Too many consecutive errors ({self._consecutive_errors}), shutting down"
            )
            return False
        
        if len(self._error_timestamps) >= self.MAX_CONSECUTIVE_ERRORS * 2:
            self.logger.critical(
                f"Too many errors in {self.ERROR_WINDOW_SECONDS}s window, shutting down"
            )
            return False
        
        return True
    
    def reset_consecutive(self) -> None:
        """Reset consecutive error counter after successful operation."""
        self._consecutive_errors = 0


# =============================================================================
# Main Entry Point
# =============================================================================

async def main() -> int:
    """
    Main entry point for VESPER Virtual Assistant.
    
    Lifecycle:
        1. Parse arguments and load configuration
        2. Initialize logging
        3. Initialize EventBus (singleton)
        4. Create Brain orchestrator
        5. Brain registers and starts all agents
        6. Run indefinitely with health monitoring
        7. Graceful shutdown on interrupt
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    global _shutdown_event, _brain_instance
    
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Load configuration
    config = load_config(args.get("config"))
    
    # Override with command line args
    if args.get("debug"):
        config.setdefault("general", {})["debug_mode"] = True
    
    # Setup logging
    setup_logging(config)
    
    # Import after logging is configured
    from utils.logger import get_logger
    from orchestrator import Brain
    from bus.event_bus import EventBus
    
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("VESPER Virtual Assistant Starting")
    logger.info(f"Python {sys.version}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("=" * 60)
    log_startup_preflight(config, logger)
    
    # Initialize error recovery
    error_recovery = ErrorRecovery(logger)
    
    # Initialize shutdown event
    _shutdown_event = asyncio.Event()
    
    # Get the event loop
    loop = asyncio.get_event_loop()
    
    # Set up signal handlers
    _create_shutdown_handler(loop, logger)
    
    # Initialize EventBus singleton
    logger.info("Initializing EventBus...")
    event_bus = EventBus()
    await event_bus.start()
    logger.info("EventBus initialized")
    
    # Create the Brain orchestrator
    logger.info("Creating Brain orchestrator...")
    brain = Brain(config=config)
    _brain_instance = brain
    
    try:
        # Start the brain (this registers and starts all agents)
        logger.info("Starting Brain and all agents...")
        logger.info("Boot sequence enabled: voice startup + full agent routing")
        await brain.start()
        logger.info("Brain started successfully")

        # PC3: optional Discord remote interface (mobile access). Guarded — it
        # no-ops unless remote.enabled AND discord.py + a token are present, so
        # it can never block or break startup for anyone not using it.
        remote_task = None
        if config.get("remote", {}).get("enabled"):
            from remote.discord_remote import run_remote
            remote_task = asyncio.create_task(run_remote(brain, config))
            logger.info("Discord remote interface enabled (remote.enabled=true)")

        # Log active agents
        logger.info("Active agents:")
        for name in brain._agents.keys():
            logger.info(f"  - {name}")
        
        # Main loop - run until shutdown is signaled
        logger.info("Entering main loop (press Ctrl+C to exit)")
        
        while not _shutdown_event.is_set():
            try:
                # Wait for shutdown signal with timeout for periodic checks
                await asyncio.wait_for(
                    _shutdown_event.wait(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                # Normal timeout - system is healthy
                error_recovery.reset_consecutive()
                
                # Periodic health check can go here
                if not brain.is_running:
                    logger.error("Brain stopped unexpectedly")
                    break
        
        # Graceful shutdown
        if remote_task is not None:
            remote_task.cancel()
        await _graceful_shutdown(brain, logger, "User interrupt")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        await _graceful_shutdown(brain, logger, "Keyboard interrupt")
        return 0
        
    except Exception as e:
        should_continue = error_recovery.record_error(e, "main loop")
        
        if not should_continue:
            logger.critical("Fatal error, system cannot recover")
            with suppress(Exception):
                await _graceful_shutdown(brain, logger, f"Fatal error: {e}")
            return 1
        
        # Try graceful shutdown even on error
        with suppress(Exception):
            await _graceful_shutdown(brain, logger, f"Error: {e}")
        return 1
        
    finally:
        logger.info("=" * 60)
        logger.info("VESPER Virtual Assistant Stopped")
        logger.info("=" * 60)


def run() -> None:
    """
    Entry point for running as a module or script.
    
    Can be invoked with:
        python main.py
        python -m vesper
    
    Handles the asyncio event loop lifecycle.
    """
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        # Handle Ctrl+C during startup
        print("\nInterrupted during startup")
        exit_code = 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)


# =============================================================================
# Cleanup Registration
# =============================================================================

@atexit.register
def _cleanup_on_exit() -> None:
    """
    Cleanup handler called when Python interpreter exits.
    
    Ensures resources are released even on unexpected exit.
    """
    global _brain_instance
    
    if _brain_instance is not None:
        # Note: atexit handlers can't run async code properly
        # The graceful shutdown in main() should handle this
        print("VESPER cleanup complete")


if __name__ == "__main__":
    run()
