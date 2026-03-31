import unittest
import json
import logging
import io
import uuid
import jsonschema
import os

from src.core.observability import FlightRecorder

class TestObservability(unittest.TestCase):
    def setUp(self):
        # Create an in-memory stream to capture logs
        self.log_stream = io.StringIO()
        self.handler = logging.StreamHandler(self.log_stream)

        # Get the specific logger used by FlightRecorder
        self.logger = logging.getLogger("FlightRecorder")
        self.logger.setLevel(logging.INFO)

        # Remove any existing handlers to avoid duplicate logs in testing
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self.handler)

        # Load the schema
        schema_path = os.path.join(
            os.path.dirname(__file__), '..', 'enterprise_bundle', 'schemas', 'audit_log.json'
        )
        with open(schema_path, 'r') as f:
            self.audit_schema = json.load(f)

    def test_flight_recorder_log_event(self):
        recorder = FlightRecorder(agent_id="test-agent-123", trace_id="test-trace-123")
        recorder.log_event("test_event", {"key": "value"})

        # Retrieve the log
        self.handler.flush()
        log_output = self.log_stream.getvalue().strip()

        # Ensure it's valid JSON
        log_entry = json.loads(log_output)

        # Validate against our new schema
        jsonschema.validate(instance=log_entry, schema=self.audit_schema)

        self.assertEqual(log_entry["agent_id"], "test-agent-123")
        self.assertEqual(log_entry["trace_id"], "test-trace-123")
        self.assertEqual(log_entry["event_type"], "test_event")
        self.assertEqual(log_entry["details"], {"key": "value"})
        self.assertIn("timestamp", log_entry)

    def test_flight_recorder_log_tool_call(self):
        recorder = FlightRecorder() # test with auto-generated UUIDs
        recorder.log_tool_call(
            tool_name="calculate_risk",
            arguments={"portfolio_value": 1000, "risk_factor": 0.5},
            result=500,
            status="success"
        )

        self.handler.flush()
        log_output = self.log_stream.getvalue().strip()
        log_entry = json.loads(log_output)

        # Validate
        jsonschema.validate(instance=log_entry, schema=self.audit_schema)

        self.assertEqual(log_entry["event_type"], "tool_call")
        self.assertEqual(log_entry["details"]["tool_name"], "calculate_risk")
        self.assertEqual(log_entry["details"]["result"], 500)
        self.assertEqual(log_entry["details"]["status"], "success")

if __name__ == '__main__':
    unittest.main()
