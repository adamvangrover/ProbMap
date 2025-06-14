import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file at the application start
load_dotenv()

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "Proprietary Probability Map API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Conceptual Vega Database URL
    VEGA_DATABASE_URL: str | None = os.getenv("VEGA_DATABASE_URL")

    # Path for model artifacts
    MODEL_ARTIFACT_PATH: str = os.getenv("MODEL_ARTIFACT_PATH", "./models_store")

    # Example: API Port
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from .env

# Instantiate settings to be imported by other modules
settings = Settings()

# Ensure model artifact path exists
if not os.path.exists(settings.MODEL_ARTIFACT_PATH):
    os.makedirs(settings.MODEL_ARTIFACT_PATH, exist_ok=True)

if __name__ == "__main__":
    # Example of how to use the settings
    print(f"App Name: {settings.APP_NAME}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"Log Level: {settings.LOG_LEVEL}")
    print(f"Vega DB URL: {settings.VEGA_DATABASE_URL}")
    print(f"Model Artifact Path: {settings.MODEL_ARTIFACT_PATH}")
    print(f"API Port: {settings.API_PORT}")
    # Try creating a .env file with some of these values to test
    # E.g. VEGA_DATABASE_URL="my_actual_db_url"
    # DEBUG=True
