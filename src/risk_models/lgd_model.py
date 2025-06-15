import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# For a more advanced LGD model, Beta Regression is common, but statsmodels is a heavy dependency.
# from statsmodels.discrete.discrete_model import BetaModel
# For PoC, we'll use a simpler approach: mean LGD per collateral type or a simple linear regression.
from sklearn.ensemble import GradientBoostingRegressor # Changed from LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from typing import Optional, Dict, Any, List
from pathlib import Path
import datetime # For model versioning

from src.core.config import settings
from src.data_management.knowledge_base import KnowledgeBaseService
# Already imported ModelRegistry in the previous diff for PDModel, but good to ensure it's here for LGDModel context
from src.mlops.model_registry import ModelRegistry


logger = logging.getLogger(__name__)

class LGDModel:
    """
    Loss Given Default (LGD) Model.
    For PoC, this is a very basic model, e.g., predicting recovery rate based on collateral type
    and then LGD = 1 - Recovery Rate.
    A more robust implementation might use Beta Regression or a more complex ML model.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(settings.MODEL_ARTIFACT_PATH) / "lgd_model.joblib"
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: Optional[Pipeline] = None # Or could be a dict for mean-based models
        self._feature_names: List[str] = []

        # For PoC, let's try a simple Linear Regression on some synthetic recovery rates
        # Features: collateral_type (OHE), loan_amount_usd (as example numeric)
        self.numerical_features = ['loan_amount_usd', 'economic_condition_indicator'] # Added economic_condition_indicator
        self.categorical_features = ['collateral_type', 'seniority_of_debt'] # Added seniority_of_debt

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), self.numerical_features),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), self.categorical_features)
            ],
            remainder='drop'
        )
        self.base_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', GradientBoostingRegressor(random_state=42, n_estimators=100))])


    def _prepare_features_and_target(self, kb_service: KnowledgeBaseService) -> pd.DataFrame:
        """
        Prepares features and a synthetic target (recovery_rate) for LGD model training.
        In a real scenario, historical recovery rates would be used.
        """
        logger.info("Preparing features for LGD model...")
        all_loans = kb_service.get_all_loans()

        records = []
        for loan in all_loans:
            # Synthetic recovery rate based on collateral for PoC
            # Synthetic recovery rate based on collateral for PoC
            base_recovery = 0.1 # Default low recovery
            if loan.collateral_type:
                collateral_value_str = str(loan.collateral_type.value) # Ensure it's a string for comparison
                if collateral_value_str == "Real Estate": base_recovery = 0.7
                elif collateral_value_str == "Equipment": base_recovery = 0.5
                elif collateral_value_str == "Receivables": base_recovery = 0.4
                elif collateral_value_str == "Inventory": base_recovery = 0.3

            recovery_rate_adjusted = base_recovery

            # Adjustment for Seniority
            seniority = str(loan.seniority_of_debt) if loan.seniority_of_debt else 'Unknown'
            if seniority == 'Senior':
                recovery_rate_adjusted += 0.10
            elif seniority == 'Subordinated':
                recovery_rate_adjusted -= 0.15

            # Adjustment for Economic Condition
            economic_indicator = loan.economic_condition_indicator if loan.economic_condition_indicator is not None else 0.5
            recovery_rate_adjustment_econ = (economic_indicator - 0.5) * 0.2 # Max +/- 0.1
            recovery_rate_adjusted += recovery_rate_adjustment_econ

            # Add some noise
            recovery_rate_final = recovery_rate_adjusted + np.random.normal(0, 0.05) # Reduced noise std dev

            # Clip the final recovery_rate
            recovery_rate_final = np.clip(recovery_rate_final, 0.05, 0.95)

            if loan.default_status: # Only consider defaulted loans for LGD typically
                record = {
                    'loan_id': loan.loan_id,
                    'loan_amount_usd': loan.loan_amount,
                    'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None',
                    'seniority_of_debt': seniority,
                    'economic_condition_indicator': economic_indicator,
                    'recovery_rate': recovery_rate_final # Target variable
                }
                records.append(record)

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No defaulted loans available to prepare LGD features, or no loans at all.")
            return pd.DataFrame()

        logger.info(f"Prepared {len(df)} records from defaulted loans for LGD model.")
        return df

    def train(self, kb_service: KnowledgeBaseService, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """Trains the LGD model."""
        logger.info(f"Starting LGD model training. Model will be saved to {self.model_path}")
        df = self._prepare_features_and_target(kb_service)

        if df.empty or 'recovery_rate' not in df.columns or len(df) < 5: # Basic check
            logger.error("Not enough data or 'recovery_rate' column missing. LGD Training aborted.")
            return {"error": "Insufficient data for LGD training."}

        X = df.drop(['recovery_rate', 'loan_id'], axis=1)
        y = df['recovery_rate']

        # Ensure all defined features are present
        for feature_set in [self.numerical_features, self.categorical_features]:
            for col in feature_set:
                if col not in X.columns:
                    X[col] = 0 if col in self.numerical_features else 'Unknown'


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        self.model = self.base_model.fit(X_train, y_train)

        try:
            ohe_feature_names = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self._feature_names = self.numerical_features + list(ohe_feature_names)
        except Exception as e:
            logger.warning(f"Could not extract LGD feature names after OHE: {e}")
            self._feature_names = X_train.columns.tolist()


        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = {
            "train_mse": mean_squared_error(y_train, y_pred_train),
            "test_mse": mean_squared_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        }
        logger.info(f"LGD Training complete. Metrics: {metrics}")
        self.save_model()

        # Automate model registration
        if "error" not in metrics and self.model: # Ensure model exists before getting params
            try:
                registry = ModelRegistry()
                model_params = self.model.named_steps['regressor'].get_params()
                model_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                registry.register_model(
                    model_name="LGDModel",
                    model_version=model_version,
                    model_path=str(self.model_path),
                    metrics=metrics,
                    parameters=model_params,
                    tags={"stage": "training", "model_type": "GradientBoostingRegressor"}
                )
                logger.info(f"LGDModel version {model_version} registered successfully.")
            except Exception as e:
                logger.error(f"Error during model registration for LGDModel: {e}")
        elif not self.model and "error" not in metrics:
            logger.warning("LGDModel training seemed successful but model object is None. Skipping registration.")


        return metrics

    def predict_lgd(self, loan_features: Dict[str, Any]) -> float:
        """
        Predicts LGD for a given set of loan features.
        Input: A dictionary with features like 'collateral_type', 'loan_amount_usd'.
        Returns LGD (a float between 0 and 1).
        LGD = 1 - RecoveryRate
        """
        if self.model is None:
            logger.error("LGD Model not trained or loaded. Call train() or load_model() first.")
            return 0.75 # Default high LGD

        # Create DataFrame from input dict
        # Ensure all features expected by the model are present

        # Provide defaults for new features if missing
        if 'seniority_of_debt' not in loan_features:
            loan_features['seniority_of_debt'] = 'Unknown'
        if 'economic_condition_indicator' not in loan_features:
            loan_features['economic_condition_indicator'] = 0.5 # Default to neutral

        # Defaulting for other features if missing
        for col in self.numerical_features:
            if col not in loan_features:
                loan_features[col] = 0 # Or np.nan and let imputer handle it
        for col in self.categorical_features:
            if col not in loan_features:
                loan_features[col] = 'None' # Or 'Unknown'

        data_df = pd.DataFrame([loan_features])

        try:
            predicted_recovery_rate = self.model.predict(data_df)[0]
            # Clamp recovery rate between 0.05 and 0.95 as LGD can't be <0 or >1
            clamped_recovery_rate = np.clip(predicted_recovery_rate, 0.05, 0.95)
            lgd = 1.0 - clamped_recovery_rate
            return float(lgd)
        except Exception as e:
            logger.error(f"Error during LGD prediction: {e}. Features: {loan_features}")
            return 0.75 # Default high LGD on error

    def save_model(self):
        if self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
                logger.info(f"LGD Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Error saving LGD model: {e}")
        else:
            logger.warning("No LGD model to save.")

    def load_model(self) -> bool:
        model_loaded_successfully = False
        # Try loading from the specific model_path first (if it exists)
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"LGD Model loaded from specified path: {self.model_path}")
                model_loaded_successfully = True
            except Exception as e:
                logger.error(f"Error loading LGD model from {self.model_path}: {e}. Trying registry.")
                self.model = None # Ensure model is None if loading fails

        if not model_loaded_successfully:
            logger.info(f"Model file not found at {self.model_path} or failed to load. Attempting to load 'production' model from registry.")
            try:
                registry = ModelRegistry()
                prod_model_path_str = registry.get_production_model_path("LGDModel")
                if prod_model_path_str:
                    prod_model_path = Path(prod_model_path_str)
                    if prod_model_path.exists():
                        self.model = joblib.load(prod_model_path)
                        self.model_path = prod_model_path # Update model_path to the one loaded
                        logger.info(f"LGD Model (production) loaded from registry path: {self.model_path}")
                        model_loaded_successfully = True
                    else:
                        logger.warning(f"Production LGD model path from registry does not exist: {prod_model_path}")
                else:
                    logger.warning("No production LGDModel found in registry.")
            except Exception as e:
                logger.error(f"Error loading production LGDModel from registry: {e}")
                self.model = None # Ensure model is None if registry loading fails

        if not model_loaded_successfully:
             logger.warning(f"LGD Model could not be loaded from specified path or registry.")

        return model_loaded_successfully

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing LGDModel ---")
    kb = KnowledgeBaseService()

    if not kb.get_all_loans(): # Check if there's any loan data at all
        logger.error("KnowledgeBaseService could not load loan data. LGD Model tests cannot proceed robustly.")
    else:
        lgd_model_instance = LGDModel()

        # Train LGD model
        logger.info("Training LGD model...")
        # Need enough defaulted loans for this to work. The _prepare_features filters for default_status = True.
        # Our sample_loans.json has one defaulted loan. This might not be enough for a robust train/test split.
        # The train_test_split might complain if only one sample after filtering.
        # For PoC, if training fails due to insufficient data, it should be logged.
        train_metrics_lgd = lgd_model_instance.train(kb_service=kb)
        logger.info(f"LGD Model training metrics: {train_metrics_lgd}")

        # After training, list models from registry
        if "error" not in train_metrics_lgd:
            registry = ModelRegistry()
            logger.info("LGD Models in registry after training:")
            lgd_models_in_registry = registry.list_models("LGDModel")
            for m in lgd_models_in_registry:
                logger.info(f"  - Version: {m['model_version']}, Path: {m['model_path']}, Status: {m.get('status')}, Metrics: {m.get('metrics')}")

            # Promote the just trained model to "production" for testing fallback loading
            if lgd_models_in_registry:
                latest_version = lgd_models_in_registry[0]['model_version'] # Assumes latest is first
                logger.info(f"Promoting LGDModel version {latest_version} to 'production' for testing.")
                registry.update_model_status("LGDModel", latest_version, "production")

        # Test fallback loading
        logger.info("--- Testing Fallback Model Loading for LGDModel ---")
        lgd_model_for_fallback_test = LGDModel(model_path=Path("models_store/non_existent_lgd_model.joblib"))
        load_success_fallback_lgd = lgd_model_for_fallback_test.load_model()
        logger.info(f"LGDModel fallback loading attempt. Success: {load_success_fallback_lgd}")
        if load_success_fallback_lgd and lgd_model_for_fallback_test.model is not None:
            logger.info(f"Fallback LGDModel loaded from: {lgd_model_for_fallback_test.model_path}")
        else:
            logger.warning("Fallback LGDModel loading failed or no production model was found.")

        logger.info("--- Resuming tests with originally trained/loaded LGDModel instance ---")
        if lgd_model_instance.model is None:
            logger.info("Re-loading original LGD model instance for subsequent tests...")
            load_success_lgd = lgd_model_instance.load_model()
            logger.info(f"Model loaded successfully for original LGD instance: {load_success_lgd}")


        if lgd_model_instance.model is not None:
            # Test prediction with some sample features
            sample_features_for_lgd_1 = {
                'collateral_type': 'Real Estate',
                'loan_amount_usd': 5000000,
                'seniority_of_debt': 'Senior',
                'economic_condition_indicator': 0.75 # Good conditions
            }
            predicted_lgd_1 = lgd_model_instance.predict_lgd(sample_features_for_lgd_1)
            logger.info(f"Predicted LGD (Real Estate, Senior, Good Econ): {predicted_lgd_1:.4f}")

            sample_features_for_lgd_2 = {
                'collateral_type': 'Inventory',
                'loan_amount_usd': 100000,
                'seniority_of_debt': 'Subordinated',
                'economic_condition_indicator': 0.25 # Bad conditions
            }
            predicted_lgd_2 = lgd_model_instance.predict_lgd(sample_features_for_lgd_2)
            logger.info(f"Predicted LGD (Inventory, Subordinated, Bad Econ): {predicted_lgd_2:.4f}")

            sample_features_for_lgd_3 = {
                'collateral_type': 'None',
                'loan_amount_usd': 200000,
                'seniority_of_debt': 'Unknown', # Test default for seniority
                'economic_condition_indicator': 0.5 # Test default for econ indicator
            }
            predicted_lgd_3 = lgd_model_instance.predict_lgd(sample_features_for_lgd_3)
            logger.info(f"Predicted LGD (None Collateral, Unknown Seniority, Neutral Econ): {predicted_lgd_3:.4f}")

            sample_features_for_lgd_4 = { # Missing new features to test defaults in predict_lgd
                'collateral_type': 'Equipment',
                'loan_amount_usd': 750000
            }
            predicted_lgd_4 = lgd_model_instance.predict_lgd(sample_features_for_lgd_4)
            logger.info(f"Predicted LGD (Equipment, Missing Seniority/Econ): {predicted_lgd_4:.4f}")
        else:
            logger.error("LGD Model could not be loaded or trained. Prediction tests skipped.")
