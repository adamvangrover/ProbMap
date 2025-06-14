import logging
import sys
from src.core.config import settings # Import settings from the config module

def setup_logging():
    """Configures logging for the application."""
    log_level = settings.LOG_LEVEL.upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s"

    # Basic configuration
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout) # Log to stdout
            # You can add FileHandler here if you want to log to a file
            # logging.FileHandler("app.log")
        ]
    )

    # Example: Silence overly verbose libraries if needed
    # logging.getLogger("some_verbose_library").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {log_level}")

# Call setup_logging when this module is imported,
# or call it explicitly in your main application entry point.
# For now, let's set it up on import.
setup_logging()

# Example usage:
if __name__ == "__main__":
    # This logger will inherit the root logger's configuration
    test_logger = logging.getLogger("TestLogger")
    test_logger.debug("This is a debug message.") # Won't show if LOG_LEVEL is INFO
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
    test_logger.critical("This is a critical message.")
