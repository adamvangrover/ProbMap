import datetime
import re

def get_current_timestamp_iso() -> str:
    """Returns the current timestamp in ISO 8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def is_valid_uuid(uuid_to_test: str, version: int = 4) -> bool:
    """
    Validate that a string is a valid UUID.
    Parameters:
    uuid_to_test (str): The string to test.
    version (int): UUID version to test against (1, 2, 3, 4, 5). Default is 4.
    Returns:
    bool: True if valid UUID, False otherwise.
    """
    if not isinstance(uuid_to_test, str): # Explicitly handle non-string inputs
        return False

    if version == 4:
        # Regex for UUID version 4:
        # - 8 hex chars
        # - 4 hex chars
        # - '4' (version bit) followed by 3 hex chars
        # - one of '8', '9', 'a', 'b' (variant bits) followed by 3 hex chars
        # - 12 hex chars
        regex = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z', re.I)
        match = regex.match(uuid_to_test)
        return bool(match)
    # Add regex for other versions if needed
    # For simplicity, only v4 is fully implemented here.
    # A more robust solution might use the 'uuid' module:
    # import uuid
    # try:
    #     uuid_obj = uuid.UUID(uuid_to_test, version=version)
    #     return str(uuid_obj) == uuid_to_test # Ensure it's not just a prefix
    # except ValueError:
    #     return False
    else: # Placeholder for other versions
        # This error should propagate if the version is not implemented.
        raise NotImplementedError(f"UUID version {version} validation not implemented.")

    # Removed the broad try-except Exception: return False
    # If re.compile or .match were to raise an unexpected error for some input,
    # it would now propagate, which is generally better for debugging.
    # The primary expected "failure" is bool(match) being False.

if __name__ == "__main__":
    print(f"Current ISO Timestamp: {get_current_timestamp_iso()}")

    valid_uuid_v4 = "123e4567-e89b-42d3-a456-426614174000"
    invalid_uuid_v4 = "123e4567-e89b-12d3-a456-426614174000" # Invalid version bit
    not_a_uuid = "hello-world"

    print(f"'{valid_uuid_v4}' is valid UUIDv4: {is_valid_uuid(valid_uuid_v4)}")
    print(f"'{invalid_uuid_v4}' is valid UUIDv4: {is_valid_uuid(invalid_uuid_v4)}")
    print(f"'{not_a_uuid}' is valid UUIDv4: {is_valid_uuid(not_a_uuid)}")
