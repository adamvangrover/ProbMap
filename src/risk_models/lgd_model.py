import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# For a more advanced LGD model, Beta Regression is common, but statsmodels is a heavy dependency.
# from statsmodels.discrete.discrete_model import BetaModel
# For PoC, we'll use a simpler approach: mean LGD per collateral type or a simple linear regression.
from sklearn.linear_model import LinearRegression # Example for a simple regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from typing import Optional, Dict, Any, List
from pathlib import Path

from src.core.config import settings
from src.data_management.knowledge_base import KnowledgeBaseService

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
        self.numerical_features = ['loan_amount_usd'] # Example numeric feature
        self.categorical_features = ['collateral_type'] # Key feature for LGD

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), self.numerical_features),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), self.categorical_features)
            ],
            remainder='drop'
        )
        self.base_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('regressor', LinearRegression())])


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
            # These are made-up values for demonstration
            recovery_rate = 0.1 # Default low recovery
            if loan.collateral_type:
                if loan.collateral_type.value == "Real Estate": recovery_rate = 0.7
                elif loan.collateral_type.value == "Equipment": recovery_rate = 0.5
                elif loan.collateral_type.value == "Receivables": recovery_rate = 0.4
                elif loan.collateral_type.value == "Inventory": recovery_rate = 0.3

            # Add some noise to make it seem more realistic for regression
            recovery_rate = np.clip(recovery_rate + np.random.normal(0, 0.1), 0.05, 0.95)

            if loan.default_status: # Only consider defaulted loans for LGD typically
                record = {
                    'loan_id': loan.loan_id,
                    'loan_amount_usd': loan.loan_amount,
                    'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None',
                    'recovery_rate': recovery_rate # Target variable
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
        for col in self.numerical_features + self.categorical_features:
            if col not in loan_features:
                 # Defaulting strategy - might need more sophisticated handling
                loan_features[col] = 0 if col in self.numerical_features else 'None'

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
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"LGD Model loaded from {self.model_path}")
                return True
            else:
                logger.warning(f"LGD Model file not found at {self.model_path}. Model not loaded.")
                return False
        except Exception as e:
            logger.error(f"Error loading LGD model: {e}")
            self.model = None
            return False

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

        # Load LGD model
        logger.info("Loading LGD model...")
        load_success_lgd = lgd_model_instance.load_model()
        logger.info(f"LGD Model loaded successfully: {load_success_lgd}")

        if load_success_lgd and lgd_model_instance.model is not None:
            # Test prediction with some sample features
            sample_features_for_lgd = {
                'collateral_type': 'Real Estate', # Value from CollateralType enum
                'loan_amount_usd': 5000000
            }
            predicted_lgd = lgd_model_instance.predict_lgd(sample_features_for_lgd)
            logger.info(f"Predicted LGD for Real Estate collateral, 5M loan: {predicted_lgd:.4f}")

            sample_features_for_lgd_2 = {
                'collateral_type': 'Inventory',
                'loan_amount_usd': 100000
            }
            predicted_lgd_2 = lgd_model_instance.predict_lgd(sample_features_for_lgd_2)
            logger.info(f"Predicted LGD for Inventory collateral, 100k loan: {predicted_lgd_2:.4f}")

            sample_features_for_lgd_3 = { # Example with a type that might be unknown if not in training's OHE
                'collateral_type': 'Spaceship', # Unknown type
                'loan_amount_usd': 100000000
            }
            predicted_lgd_3 = lgd_model_instance.predict_lgd(sample_features_for_lgd_3)
            logger.info(f"Predicted LGD for 'Spaceship' collateral: {predicted_lgd_3:.4f}")
        else:
            logger.error("LGD Model could not be loaded or trained. Prediction tests skipped.")
