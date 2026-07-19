"""
Tool Executor - Async execution engine for Jarvis tools.

This module handles:
- Permission checking before execution
- Emitting ActionRequestEvent to target agents
- Tracking execution results via ActionResultEvent
- Timeout handling
- Observable execution (events for start/complete/error)

Key Design:
- Tools NEVER execute directly - they emit events
- Executions are validated and routed through the EventBus
- All execution is async and observable
- Results are tracked via correlation IDs
- Permission checks happen before event emission
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from bus.event_bus import EventBus, get_event_bus
from schemas.events import (
    ActionRequestEvent,
    ActionResultEvent,
)
from tools.registry import (
    ToolDefinition,
    ToolPermission,
    ToolRegistry,
    get_tool_registry,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Execution Status
# =============================================================================

class ToolExecutionStatus(Enum):
    """Status of a tool execution."""
    
    PENDING = auto()      # Waiting to start
    VALIDATING = auto()   # Validating args and permissions
    EXECUTING = auto()    # Event emitted, waiting for result
    COMPLETED = auto()    # Successfully completed
    FAILED = auto()       # Failed with error
    TIMEOUT = auto()      # Timed out waiting for result
    CANCELLED = auto()    # Cancelled by user/system
    DENIED = auto()       # Permission denied


# =============================================================================
# Execution Result
# =============================================================================

@dataclass
class ToolExecutionResult:
    """
    Result of a tool execution.
    
    Attributes:
        invocation_id: Unique ID for this execution
        tool_name: Name of the tool that was executed
        status: Final status of the execution
        result: Result data (if successful)
        error: Error message (if failed)
        start_time: When execution started
        end_time: When execution completed
        execution_time_ms: Total execution time in milliseconds
        metadata: Additional metadata about the execution
    """
    
    invocation_id: UUID
    tool_name: str
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    result: Any = None
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolExecutionStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invocation_id": str(self.invocation_id),
            "tool_name": self.tool_name,
            "status": self.status.name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


# =============================================================================
# Permission Checker
# =============================================================================

class ToolPermissionChecker:
    """
    Checks permissions before tool execution.
    
    This class enforces permission policies:
    - Blocked tools cannot be executed
    - High-risk tools may require confirmation
    - Allowed tools list filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._require_confirmation_levels: Set[ToolPermission] = {
            ToolPermission.HIGH_RISK,
            ToolPermission.CRITICAL,
        }
        self._blocked_tools: Set[str] = set()
        self._allowed_tools: Set[str] = set()
        
        # Load from config
        safety_config = self._config.get("safety", {})
        blocked_ops = safety_config.get("blocked_operations", [])
        self._blocked_tools.update(blocked_ops)
    
    def check_permission(
        self,
        tool: ToolDefinition,
        args: Dict[str, Any],
    ) -> tuple[bool, str]:
        """
        Check if a tool can be executed with given args.
        
        Args:
            tool: The tool definition
            args: Arguments for the tool
        
        Returns:
            Tuple of (allowed, reason)
        """
        # Check if tool is blocked
        if tool.name in self._blocked_tools:
            return False, f"Tool '{tool.name}' is blocked"
        
        if tool.permission == ToolPermission.BLOCKED:
            return False, f"Tool '{tool.name}' has BLOCKED permission level"
        
        # Check allowed list (if set)
        if self._allowed_tools and tool.name not in self._allowed_tools:
            return False, f"Tool '{tool.name}' is not in allowed list"
        
        # Check for confirmation required (but don't block)
        if tool.permission in self._require_confirmation_levels:
            logger.info(
                f"Tool '{tool.name}' requires confirmation "
                f"(permission={tool.permission.value})"
            )
        
        return True, "Allowed"
    
    def requires_confirmation(self, tool: ToolDefinition) -> bool:
        """Check if a tool requires user confirmation."""
        return tool.permission in self._require_confirmation_levels
    
    def set_allowed_tools(self, tools: List[str]) -> None:
        """Set the list of allowed tools."""
        self._allowed_tools = set(tools)
    
    def block_tool(self, name: str) -> None:
        """Block a tool."""
        self._blocked_tools.add(name)
    
    def unblock_tool(self, name: str) -> None:
        """Unblock a tool."""
        self._blocked_tools.discard(name)


# =============================================================================
# Tool Executor
# =============================================================================

class ToolExecutor:
    """
    Async execution engine for Jarvis tools.
    
    The executor:
    - Validates tool existence and arguments
    - Validates tool permissions before execution
    - Emits ActionRequestEvent to target agents
    - Tracks pending executions
    - Waits for ActionResultEvent responses
    - Handles timeouts and errors
    
    Usage:
        executor = get_tool_executor()
        result = await executor.execute("open_app", {"app_name": "Safari"})
        if result.success:
            print(f"Opened app: {result.result}")
    """
    
    _instance: Optional[ToolExecutor] = None
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._event_bus = event_bus or get_event_bus()
        self._registry = registry or get_tool_registry()
        self._config = config or {}
        self._permission_checker = ToolPermissionChecker(config)
        
        
        # Track pending executions
        self._pending: Dict[UUID, asyncio.Future] = {}
        self._results: Dict[UUID, ToolExecutionResult] = {}
        
        # Subscribe to result events
        self._subscribed = False
    
    @classmethod
    def get_instance(
        cls,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutor:
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(event_bus=event_bus, config=config)
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize the executor and subscribe to events."""
        if self._subscribed:
            return
        
        # Subscribe to ActionResultEvent to track completions
        self._event_bus.subscribe(ActionResultEvent, self._handle_result)
        self._subscribed = True
        logger.info("Tool executor initialized")
    
    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        correlation_id: Optional[UUID] = None,
        plan_id: Optional[UUID] = None,
        timeout: Optional[float] = None,
        emit_tool_call_event: bool = True,
        source: str = "unknown",
    ) -> ToolExecutionResult:
        """
        Execute a tool by name with given arguments.
        
        This method:
        1. Validates the tool exists
        2. Validates arguments
    3. Checks execution policy
    4. Emits ActionRequestEvent
    5. Waits for ActionResultEvent
    6. Returns the result
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            correlation_id: Optional correlation ID for tracking
            plan_id: Optional plan ID if part of a plan
            timeout: Optional timeout override (uses tool default if not set)
            emit_tool_call_event: Whether to emit tool call events
            source: Who is requesting the execution
        
        Returns:
            ToolExecutionResult with status and result/error
        """
        invocation_id = uuid4()
        start_time = datetime.now(timezone.utc)
        
        result = ToolExecutionResult(
            invocation_id=invocation_id,
            tool_name=tool_name,
            status=ToolExecutionStatus.VALIDATING,
            start_time=start_time,
        )
        
        try:
            # Step 1: Get tool definition
            tool = self._registry.get_tool(tool_name)
            if not tool:
                result.status = ToolExecutionStatus.FAILED
                result.error = f"Unknown tool: {tool_name}"
                return self._finalize_result(result)
            
            # Step 2: Validate arguments
            validation_errors = tool.validate_args(args)
            if validation_errors:
                result.status = ToolExecutionStatus.FAILED
                result.error = f"Validation failed: {'; '.join(validation_errors)}"
                return self._finalize_result(result)
            
            # Step 3: Check tool permission via local policy
            allowed, reason = self._permission_checker.check_permission(tool, args)
            if not allowed:
                result.status = ToolExecutionStatus.DENIED
                result.error = reason
                result.metadata["denial_type"] = "policy"
                return self._finalize_result(result)
            
            # Step 5: Create future for result tracking
            result_future: asyncio.Future = asyncio.Future()
            self._pending[invocation_id] = result_future
            
            # Step 6: Emit ActionRequestEvent
            result.status = ToolExecutionStatus.EXECUTING
            
            await self._event_bus.emit(ActionRequestEvent(
                action=tool.action,
                target_agent=tool.target_agent,
                parameters=args,
                correlation_id=correlation_id,
                source="ToolExecutor",
            ))
            
            logger.info(
                f"Tool execution started: {tool_name} -> {tool.target_agent}.{tool.action}"
            )
            
            # Step 7: Wait for result with timeout
            effective_timeout = timeout or tool.timeout_seconds
            try:
                action_result = await asyncio.wait_for(
                    result_future,
                    timeout=effective_timeout,
                )
                
                # Process the ActionResultEvent
                if action_result.success:
                    result.status = ToolExecutionStatus.COMPLETED
                    result.result = action_result.result
                else:
                    result.status = ToolExecutionStatus.FAILED
                    result.error = action_result.error or "Action failed"
                
            except asyncio.TimeoutError:
                result.status = ToolExecutionStatus.TIMEOUT
                result.error = f"Timeout after {effective_timeout}s"
            
            except asyncio.CancelledError:
                result.status = ToolExecutionStatus.CANCELLED
                result.error = "Execution cancelled"
            
            finally:
                # Cleanup pending
                self._pending.pop(invocation_id, None)
            
        except Exception as e:
            result.status = ToolExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Tool execution error: {e}", exc_info=True)
        
        return self._finalize_result(result)
    
    async def execute_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        parallel: bool = False,
        stop_on_error: bool = True,
    ) -> List[ToolExecutionResult]:
        """
        Execute multiple tools in sequence or parallel.
        
        Args:
            tool_calls: List of {"tool_name": str, "args": dict}
            parallel: Whether to execute in parallel
            stop_on_error: Whether to stop on first error (sequential only)
        
        Returns:
            List of execution results
        """
        if parallel:
            # Execute all in parallel
            tasks = [
                self.execute(call["tool_name"], call.get("args", {}))
                for call in tool_calls
            ]
            return await asyncio.gather(*tasks)
        else:
            # Execute sequentially
            results = []
            for call in tool_calls:
                result = await self.execute(call["tool_name"], call.get("args", {}))
                results.append(result)
                
                if stop_on_error and not result.success:
                    break
            
            return results
    
    def cancel(self, invocation_id: UUID) -> bool:
        """
        Cancel a pending execution.
        
        Args:
            invocation_id: ID of the execution to cancel
        
        Returns:
            True if execution was found and cancelled
        """
        future = self._pending.get(invocation_id)
        if future and not future.done():
            future.cancel()
            return True
        return False
    
    def cancel_all(self) -> int:
        """Cancel all pending executions. Returns count cancelled."""
        count = 0
        for future in self._pending.values():
            if not future.done():
                future.cancel()
                count += 1
        return count
    
    def get_pending_count(self) -> int:
        """Get the number of pending executions."""
        return len(self._pending)
    
    def list_available_tools(
        self,
        as_openai_functions: bool = False,
    ) -> List[Any]:
        """
        List all available tools.
        
        Args:
            as_openai_functions: If True, return in OpenAI function format
        
        Returns:
            List of tool definitions or OpenAI function schemas
        """
        if as_openai_functions:
            return self._registry.to_openai_functions()
        return self._registry.list_tools()
    
    async def _handle_result(self, event: ActionResultEvent) -> None:
        """Handle ActionResultEvent to complete pending executions."""
        # Try to match by correlation_id or action
        # This is simplified - in production, you'd want better matching
        for invocation_id, future in list(self._pending.items()):
            if not future.done():
                # For now, complete the first pending future
                # TODO: Better correlation matching
                future.set_result(event)
                break
    
    def _finalize_result(self, result: ToolExecutionResult) -> ToolExecutionResult:
        """Finalize a result with timing information."""
        result.end_time = datetime.now(timezone.utc)
        if result.start_time:
            delta = result.end_time - result.start_time
            result.execution_time_ms = delta.total_seconds() * 1000
        
        self._results[result.invocation_id] = result
        
        logger.info(
            f"Tool execution complete: {result.tool_name} "
            f"status={result.status.name} time={result.execution_time_ms:.1f}ms"
        )
        
        return result
    

# =============================================================================
# Module-level singleton accessor
# =============================================================================

_executor_instance: Optional[ToolExecutor] = None


def get_tool_executor(
    event_bus: Optional[EventBus] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ToolExecutor:
    """Get the global tool executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = ToolExecutor.get_instance(event_bus, config)
    return _executor_instance
