import unittest
import json
import logging
import io
import uuid
from fastapi.testclient import TestClient
from src.mcp.tools import ToolRegistry, registry
from src.mcp.server import app

class TestMCP(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        # Clear the global registry for a clean slate, then re-register dummy
        registry._tools = {}

        def dummy_tool(x: int, y: int) -> int:
            return x + y

        def error_tool() -> str:
            raise ValueError("Intentional error")

        registry.register("dummy_tool", dummy_tool)
        registry.register("error_tool", error_tool)

    def test_tool_registry_register_and_execute(self):
        temp_registry = ToolRegistry()

        def multiply(a: int, b: int) -> int:
            return a * b

        temp_registry.register("multiply", multiply)

        # Test successful execution
        result = temp_registry.execute("multiply", a=3, b=4)
        self.assertEqual(result, 12)

        # Test executing unregistered tool
        with self.assertRaises(ValueError):
            temp_registry.execute("unknown_tool")

        # Test registering duplicate tool
        with self.assertRaises(ValueError):
            temp_registry.register("multiply", multiply)

    def test_mcp_server_success(self):
        request_data = {
            "tool_name": "dummy_tool",
            "arguments": {"x": 5, "y": 7},
            "agent_id": "agent-test",
            "trace_id": "trace-test"
        }

        response = self.client.post("/mcp/tools/call", json=request_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["result"], 12)
        self.assertIsNone(data["error"])

    def test_mcp_server_tool_not_found(self):
        request_data = {
            "tool_name": "missing_tool",
            "arguments": {}
        }

        response = self.client.post("/mcp/tools/call", json=request_data)

        # Fast API handles the ValueError from registry.execute
        self.assertEqual(response.status_code, 400)
        self.assertIn("not registered", response.json()["detail"])

    def test_mcp_server_tool_execution_error(self):
        request_data = {
            "tool_name": "error_tool",
            "arguments": {}
        }

        response = self.client.post("/mcp/tools/call", json=request_data)

        # Our MCP server catches execution errors and returns an error response object
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Intentional error", data["error"])

if __name__ == '__main__':
    unittest.main()
