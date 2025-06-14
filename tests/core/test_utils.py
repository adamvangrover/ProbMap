import unittest
import datetime
import re # Needed for direct regex usage if not using uuid module

from src.core.utils import get_current_timestamp_iso, is_valid_uuid

class TestCoreUtils(unittest.TestCase):

    def test_get_current_timestamp_iso(self):
        timestamp_str = get_current_timestamp_iso()
        # Check if it's a string
        self.assertIsInstance(timestamp_str, str)
        # Check if it matches ISO 8601 format (basic check)
        # Example: 2023-10-27T10:30:00.123456+00:00 or 2023-10-27T10:30:00.123456Z
        try:
            parsed_time = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            self.assertIsNotNone(parsed_time)
            # Ensure it's timezone-aware and UTC
            self.assertIsNotNone(parsed_time.tzinfo)
            self.assertEqual(parsed_time.tzinfo, datetime.timezone.utc)
        except ValueError:
            self.fail(f"Timestamp '{timestamp_str}' is not in valid ISO 8601 format.")

    def test_is_valid_uuid_v4(self):
        # Valid UUIDs v4
        self.assertTrue(is_valid_uuid("123e4567-e89b-42d3-a456-426614174000"))
        self.assertTrue(is_valid_uuid("abcdefab-1234-4abc-a1b2-abcdef123456"))

        # Invalid UUIDs v4
        self.assertFalse(is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")) # Invalid version '1'
        self.assertFalse(is_valid_uuid("123e4567-e89b-42d3-f456-426614174000")) # Invalid variant 'f'
        self.assertFalse(is_valid_uuid("not-a-uuid"))
        self.assertFalse(is_valid_uuid("123e4567-e89b-42d3-a456-42661417400")) # Too short
        self.assertFalse(is_valid_uuid("123e4567-e89b-42d3-a456-4266141740000"))# Too long
        self.assertFalse(is_valid_uuid("")) # Empty string
        self.assertFalse(is_valid_uuid(None)) # type error handled by regex

    def test_is_valid_uuid_other_versions_not_implemented(self):
        # Test that other versions raise NotImplementedError as per current utils.py
        with self.assertRaises(NotImplementedError):
            is_valid_uuid("some-string", version=1)
        with self.assertRaises(NotImplementedError):
            is_valid_uuid("some-string", version=3)
        with self.assertRaises(NotImplementedError):
            is_valid_uuid("some-string", version=5)

if __name__ == '__main__':
    # This allows running the tests directly: python -m tests.core.test_utils
    # However, it's better to use `python -m unittest discover tests` from the root directory.
    unittest.main()
