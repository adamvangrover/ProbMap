import json
import yaml
import os
import sys
import unittest

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enterprise_bundle.governance.adr_controls import audit_citation_density, audit_financial_math

class TestEnterpriseBundle(unittest.TestCase):

    def setUp(self):
        self.bundle_dir = os.path.join(os.path.dirname(__file__), '..', 'enterprise_bundle')

    def test_manifest_structure(self):
        manifest_path = os.path.join(self.bundle_dir, 'manifest.yaml')
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)

        self.assertIn('package', manifest)
        self.assertEqual(manifest['package'], 'adam-sovereign-credit')
        self.assertIn('version', manifest)
        self.assertIn('security_constraints', manifest)
        self.assertIn('dependencies', manifest)

    def test_manifest_security_constraints(self):
        manifest_path = os.path.join(self.bundle_dir, 'manifest.yaml')
        with open(manifest_path, 'r') as f:
            manifest = yaml.safe_load(f)

        constraints = manifest.get('security_constraints', {})
        self.assertFalse(constraints.get('allow_public_internet', True))
        self.assertEqual(constraints.get('pii_redaction_level'), 'strict')
        self.assertEqual(constraints.get('audit_logging'), 'synchronous')

    def test_schemas_validity(self):
        schemas_dir = os.path.join(self.bundle_dir, 'schemas')
        for filename in os.listdir(schemas_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(schemas_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        schema = json.load(f)
                        self.assertIsInstance(schema, dict)
                    except json.JSONDecodeError:
                        self.fail(f"Failed to parse JSON: {filename}")

    def test_sovereign_chunk_validation(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema library not installed")

        schema_path = os.path.join(self.bundle_dir, 'schemas', 'sovereign_chunk.json')
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        valid_chunk = {
            "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
            "doc_id": "987e6543-e21b-34d5-c678-426614174000",
            "text": "The borrower's EBITDA declined by 15%.",
            "embedding_vector": [0.1] * 1536,
            "spatial_context": {
                "page_number": 1,
                "bounding_box": [10.5, 20.0, 100.0, 50.5]
            }
        }

        # Should not raise an exception
        jsonschema.validate(instance=valid_chunk, schema=schema)

    def test_agents_yaml_validity(self):
        agents_dir = os.path.join(self.bundle_dir, 'agents')
        for filename in os.listdir(agents_dir):
            if filename.endswith('.yaml'):
                filepath = os.path.join(agents_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        agent_config = yaml.safe_load(f)
                        # Only validate content if it's not a placeholder comment
                        if agent_config is not None:
                            self.assertIsInstance(agent_config, dict)
                    except yaml.YAMLError:
                        self.fail(f"Failed to parse YAML: {filename}")

    def test_adr_controls_citation_density(self):
        # Test audit_citation_density
        text_pass = "This is a claim [doc_1:chunk_1]. This is another claim [doc_1:chunk_2]."
        text_fail = "This is a claim. This is another claim."

        self.assertEqual(audit_citation_density(text_pass), (True, "Passed"))
        self.assertEqual(audit_citation_density(text_fail), (False, "Insufficient Evidence Linking"))

    def test_adr_controls_financial_math(self):
        # Test audit_financial_math
        spread_pass = json.dumps({"total_assets": 100, "total_liabilities": 50, "total_equity": 50})
        spread_fail = json.dumps({"total_assets": 100, "total_liabilities": 50, "total_equity": 40})

        self.assertEqual(audit_financial_math(spread_pass), (True, "Passed"))
        self.assertFalse(audit_financial_math(spread_fail)[0])

if __name__ == '__main__':
    unittest.main()
