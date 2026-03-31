from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from src.mcp.tools import registry
from src.core.observability import FlightRecorder

app = FastAPI(title="Model Context Protocol Server")

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    agent_id: str | None = None
    trace_id: str | None = None

class ToolCallResponse(BaseModel):
    status: str
    result: Any | None = None
    error: str | None = None

@app.post("/mcp/tools/call", response_model=ToolCallResponse)
def call_tool(request: ToolCallRequest):
    """
    Executes a tool deterministically, providing observability via FlightRecorder.
    """
    recorder = FlightRecorder(agent_id=request.agent_id, trace_id=request.trace_id)

    try:
        # Observability: Start step
        recorder.log_event("tool_call_started", details={"tool_name": request.tool_name, "arguments": request.arguments})

        # Execute Deterministically
        result = registry.execute(request.tool_name, **request.arguments)

        # Determine status
        status = "error" if isinstance(result, dict) and "error" in result else "success"

        # Observability: End step
        recorder.log_tool_call(
            tool_name=request.tool_name,
            arguments=request.arguments,
            result=result,
            status=status
        )

        if status == "error":
            return ToolCallResponse(status=status, error=result["error"])

        return ToolCallResponse(status=status, result=result)

    except ValueError as e:
        error_msg = str(e)
        recorder.log_tool_call(
            tool_name=request.tool_name,
            arguments=request.arguments,
            result={"error": error_msg},
            status="error"
        )
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected execution error: {str(e)}"
        recorder.log_tool_call(
            tool_name=request.tool_name,
            arguments=request.arguments,
            result={"error": error_msg},
            status="error"
        )
        raise HTTPException(status_code=500, detail=error_msg)
