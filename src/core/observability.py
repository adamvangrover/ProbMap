import logging
import uuid
import json
from datetime import datetime, timezone
from typing import Any

class FlightRecorder:
    """
    FlightRecorder provides a robust human and machine-readable step-by-step
    audit log for observability, lineage, and provenance tracing.
    """
    def __init__(self, agent_id=None, trace_id=None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.trace_id = trace_id or str(uuid.uuid4())
        self.logger = logging.getLogger("FlightRecorder")

    def log_event(self, event_type: str, details: dict):
        """
        Logs a general event with deterministic timestamps and trace IDs.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "event_type": event_type,
            "details": details
        }
        self.logger.info(json.dumps(log_entry))

    def log_tool_call(self, tool_name: str, arguments: dict, result: Any = None, status: str = "success"):
        """
        Logs a specific tool call event for execution layer observability.
        """
        self.log_event(
            event_type="tool_call",
            details={
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "status": status
            }
        )
