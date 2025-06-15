import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import datetime

from src.core.config import settings # For a potential registry storage path

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Conceptual placeholder for a Model Registry.
    In a real MLOps pipeline, this would interact with tools like MLflow, Vertex AI Model Registry, etc.
    For PoC, this might simulate storing model metadata in a JSON file.
    """
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path(settings.MODEL_ARTIFACT_PATH) / "model_registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, List[Dict[str, Any]]] = self._load_registry()
        logger.info(f"ModelRegistry initialized. Registry path: {self.registry_path}")

    def _load_registry(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from registry file: {self.registry_path}. Initializing empty registry.")
                return {}
            except Exception as e:
                logger.error(f"Error loading model registry: {e}. Initializing empty registry.")
                return {}
        return {}

    def _save_registry(self):
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self._models, f, indent=4, default=str) # Use default=str for datetime or other non-serializable
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def register_model(self,
                       model_name: str,
                       model_version: str,
                       model_path: str, # Path to the serialized model file (e.g., joblib)
                       metrics: Dict[str, Any],
                       parameters: Optional[Dict[str, Any]] = None,
                       tags: Optional[Dict[str, str]] = None,
                       source_code_version: Optional[str] = None # e.g., Git commit hash
                       ) -> Dict[str, Any]:
        """
        Registers a new model version.
        In a real system, this would involve more robust versioning and storage.
        """
        logger.info(f"Registering model: {model_name}, version: {model_version}")
        if model_name not in self._models:
            self._models[model_name] = []

        # Check if version already exists
        for existing_model in self._models[model_name]:
            if existing_model["model_version"] == model_version:
                logger.warning(f"Model version {model_version} for {model_name} already exists. Overwriting.")
                self._models[model_name].remove(existing_model)
                break

        registration_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

        model_entry = {
            "model_name": model_name,
            "model_version": model_version,
            "model_path": str(model_path), # Store as string
            "metrics": metrics,
            "parameters": parameters or {},
            "tags": tags or {},
            "source_code_version": source_code_version or "N/A",
            "registration_timestamp": registration_time,
            "status": "registered" # Could be 'staging', 'production', 'archived'
        }

        self._models[model_name].append(model_entry)
        # Sort by registration time or version (semantic versioning ideally)
        self._models[model_name].sort(key=lambda x: x["registration_timestamp"], reverse=True)

        self._save_registry()
        logger.info(f"Model {model_name} version {model_version} registered successfully.")
        return model_entry

    def get_model_details(self, model_name: str, model_version: str) -> Optional[Dict[str, Any]]:
        """Retrieves details for a specific model version."""
        if model_name in self._models:
            for model_entry in self._models[model_name]:
                if model_entry["model_version"] == model_version:
                    return model_entry
        logger.warning(f"Model {model_name} version {model_version} not found in registry.")
        return None

    def get_latest_model(self, model_name: str, status: Optional[str] = "registered") -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest registered model for a given name and status.
        'Latest' here is based on registration timestamp or version sorting.
        """
        if model_name in self._models and self._models[model_name]:
            for model_entry in self._models[model_name]: # Assumes sorted by latest first
                if status is None or model_entry.get("status") == status:
                    logger.info(f"Retrieved latest model for {model_name} (status: {status}): version {model_entry['model_version']}")
                    return model_entry
        logger.warning(f"No model found for {model_name} with status {status}.")
        return None

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all versions of a specific model, or all models if model_name is None."""
        if model_name:
            return self._models.get(model_name, [])

        all_model_entries = []
        for name in self._models:
            all_model_entries.extend(self._models[name])
        return all_model_entries

    def update_model_status(self, model_name: str, model_version: str, new_status: str) -> bool:
        """Updates the status of a model version (e.g., 'staging', 'production', 'archived')."""
        model_entry = self.get_model_details(model_name, model_version)
        if model_entry:
            logger.info(f"Updating status for model {model_name} v{model_version} from {model_entry['status']} to {new_status}")
            model_entry['status'] = new_status
            self._save_registry()
            return True
        logger.warning(f"Cannot update status. Model {model_name} v{model_version} not found.")
        return False

    def get_production_model_path(self, model_name: str) -> Optional[str]:
        """
        Retrieves the model_path for the latest model marked as 'production'.
        """
        prod_model_entry = self.get_latest_model(model_name, status="production")
        if prod_model_entry:
            return str(prod_model_entry.get("model_path"))
        logger.info(f"No production model path found for {model_name}.")
        return None


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing ModelRegistry ---")
    # Use a temporary registry for testing
    test_registry_path = Path(settings.MODEL_ARTIFACT_PATH) / "test_model_registry.json"
    if test_registry_path.exists():
        test_registry_path.unlink() # Clean up previous test runs

    registry = ModelRegistry(registry_path=test_registry_path)

    # Register a PD model
    pd_metrics = {"accuracy": 0.85, "roc_auc": 0.92}
    pd_params = {"solver": "liblinear", "C": 1.0}
    registry.register_model("PDModel", "1.0.0", "./models_store/pd_model_v1.0.0.joblib", pd_metrics, pd_params, source_code_version="git_hash_1")

    # Register another version of PD model
    pd_metrics_v2 = {"accuracy": 0.87, "roc_auc": 0.93}
    registry.register_model("PDModel", "1.1.0", "./models_store/pd_model_v1.1.0.joblib", pd_metrics_v2, pd_params, source_code_version="git_hash_2")

    # Register an LGD model
    lgd_metrics = {"mse": 0.05, "rmse": 0.22}
    registry.register_model("LGDModel", "1.0.0", "./models_store/lgd_model_v1.0.0.joblib", lgd_metrics)

    # List models
    logger.info("\nAll registered PD models:")
    for m in registry.list_models("PDModel"):
        logger.info(f"  - Version: {m['model_version']}, Path: {m['model_path']}, Metrics: {m['metrics']}")

    # Get latest PD model
    latest_pd = registry.get_latest_model("PDModel")
    if latest_pd:
        logger.info(f"\nLatest PD Model: Version {latest_pd['model_version']}, Registered: {latest_pd['registration_timestamp']}")

    # Get specific LGD model details
    lgd_v1 = registry.get_model_details("LGDModel", "1.0.0")
    if lgd_v1:
        logger.info(f"\nLGD Model v1.0.0 details: {lgd_v1['metrics']}")

    # Update status
    registry.update_model_status("PDModel", "1.1.0", "production")
    latest_pd_prod = registry.get_latest_model("PDModel", status="production")
    if latest_pd_prod:
        logger.info(f"\nLatest PRODUCTION PD Model: Version {latest_pd_prod['model_version']}, Status: {latest_pd_prod['status']}")

    # Clean up test registry file
    if test_registry_path.exists():
        test_registry_path.unlink()
        logger.info(f"\nCleaned up test registry file: {test_registry_path}")
