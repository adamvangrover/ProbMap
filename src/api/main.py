import logging
from fastapi import FastAPI
from src.core.config import settings
from src.api import endpoints as api_endpoints # Import the router
# from src.core.logging_config import setup_logging # Already called on import in logging_config.py

logger = logging.getLogger(__name__)

# Initialize logging (if not already done by importing logging_config)
# setup_logging() # Ensures logging is configured based on settings

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="API for the Proprietary Probability Map System (PoC)",
    debug=settings.DEBUG,
)

# Include routers
app.include_router(api_endpoints.router, prefix="/api/v1", tags=["Credit Risk Services"])

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up {settings.APP_NAME} API...")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Log level: {settings.LOG_LEVEL}")
    # Here you could initialize global resources if needed, e.g., DB connections
    # For PoC, our services (KB, models) are initialized when endpoints module is loaded or on first use.

@app.on_event("shutdown")
async def shutdown_event():
    logger.info(f"Shutting down {settings.APP_NAME} API...")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {settings.APP_NAME} API. Visit /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    # This is for running the app directly with Uvicorn for development
    # Production deployments would use Gunicorn + Uvicorn workers or similar.
    logger.info(f"Starting Uvicorn server on port {settings.API_PORT} for {settings.APP_NAME}")
    uvicorn.run(app, host="0.0.0.0", port=settings.API_PORT) #, log_level=settings.LOG_LEVEL.lower())
    # The log_level for uvicorn itself can also be set here.
    # FastAPI/app logging is handled by our logging_config.
