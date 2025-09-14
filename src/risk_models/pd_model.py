import logging
import joblib
import pandas as pd
import numpy as np # Added for SHAP
import shap # Added for SHAP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # Changed from LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Any, Optional, Dict
from pathlib import Path
import datetime # For model versioning

from src.core.config import settings # For model artifact path
from src.data_management.knowledge_base import KnowledgeBaseService # To get data for training/prediction
from src.mlops.model_registry import ModelRegistry # For model registration

logger = logging.getLogger(__name__)

class PDModel:
    """
    Probability of Default (PD) Model.
    For PoC, this uses a simple Logistic Regression model.
    Features are derived from company and loan data.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(settings.MODEL_ARTIFACT_PATH) / "pd_model.joblib"
        self.model_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self.model: Optional[Pipeline] = None
        self._feature_names: List[str] = []

        # Define feature engineering and preprocessing steps
        # These are example features. Real-world features would be more complex.
        # Updated list of numerical features that will be fed into the model (some raw, some engineered)
        self.numerical_features = [
            'loan_amount_usd', 'interest_rate_percentage', 'loan_duration_days',
            'company_age_at_origination', 'debt_to_equity_ratio', 'current_ratio',
            'net_profit_margin', 'roe', 'loan_amount_x_interest_rate'
        ]
        self.categorical_features = ['industry_sector', 'collateral_type'] # company_country_iso_code could be another

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), self.numerical_features),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), self.categorical_features)
            ],
            remainder='drop' # or 'passthrough'
        )
        self.base_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))])

    def _prepare_features(self, kb_service: KnowledgeBaseService) -> pd.DataFrame:
        """
        Prepares a DataFrame suitable for training or prediction from KnowledgeBase data.
        This is a simplified feature engineering process for PoC.
        """
        logger.info("Preparing features for PD model...")
        all_loans = kb_service.get_all_loans()
        all_companies = {comp.company_id: comp for comp in kb_service.get_all_companies()}

        records = []
        for loan in all_loans:
            company = all_companies.get(loan.company_id)
            if not company:
                logger.warning(f"Company {loan.company_id} not found for loan {loan.loan_id}. Skipping.")
                continue

            # Time-based features
            loan_duration_days = (loan.maturity_date - loan.origination_date).days if loan.maturity_date and loan.origination_date else -1

            company_age_at_origination = -1
            if company.founded_date and loan.origination_date:
                if isinstance(company.founded_date, str): # ensure date object
                    try: company.founded_date = pd.to_datetime(company.founded_date).date()
                    except: company.founded_date = None
                if isinstance(loan.origination_date, str): # ensure date object
                    try: loan.origination_date = pd.to_datetime(loan.origination_date).date()
                    except: loan.origination_date = None

                if company.founded_date and loan.origination_date: # re-check after potential conversion
                    company_age_at_origination = (loan.origination_date - company.founded_date).days / 365.25

            # Financial Ratios from latest financial statement
            debt_to_equity_ratio = np.nan
            current_ratio = np.nan
            net_profit_margin = np.nan
            roe = np.nan

            statements = kb_service.get_financial_statements_for_company(company.company_id)
            if statements:
                # Sort by statement_date to get the latest
                statements.sort(key=lambda s: s.statement_date, reverse=True)
                latest_fs = statements[0]

                if latest_fs.total_liabilities_usd is not None and latest_fs.net_equity_usd is not None and latest_fs.net_equity_usd != 0:
                    debt_to_equity_ratio = latest_fs.total_liabilities_usd / latest_fs.net_equity_usd

                if latest_fs.current_assets is not None and latest_fs.current_liabilities is not None and latest_fs.current_liabilities != 0:
                    current_ratio = latest_fs.current_assets / latest_fs.current_liabilities

                if latest_fs.net_income is not None and latest_fs.revenue is not None and latest_fs.revenue != 0:
                    net_profit_margin = latest_fs.net_income / latest_fs.revenue

                if latest_fs.net_income is not None and latest_fs.net_equity_usd is not None and latest_fs.net_equity_usd != 0:
                    roe = latest_fs.net_income / latest_fs.net_equity_usd

            # Interaction term
            loan_amount_x_interest_rate = loan.loan_amount * loan.interest_rate_percentage

            record = {
                'loan_id': loan.loan_id,
                'company_id': loan.company_id,
                'loan_amount_usd': loan.loan_amount,
                'interest_rate_percentage': loan.interest_rate_percentage,
                'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None',
                'industry_sector': company.industry_sector.value if company.industry_sector else 'Other',
                # New engineered features
                'loan_duration_days': loan_duration_days,
                'company_age_at_origination': company_age_at_origination,
                'debt_to_equity_ratio': debt_to_equity_ratio,
                'current_ratio': current_ratio,
                'net_profit_margin': net_profit_margin,
                'roe': roe,
                'loan_amount_x_interest_rate': loan_amount_x_interest_rate,
                # Target variable
                'default_status': int(loan.default_status)
            }
            records.append(record)

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No data available to prepare features.")
            return pd.DataFrame()

        logger.info(f"Prepared {len(df)} records for PD model.")
        return df

    def train(self, kb_service: KnowledgeBaseService, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Trains the PD model using data from the KnowledgeBaseService.
        Returns a dictionary of performance metrics.
        """
        logger.info(f"Starting PD model training. Model will be saved to {self.model_path}")
        df = self._prepare_features(kb_service)

        if df.empty or 'default_status' not in df.columns or len(df) < 10: # Basic check for sufficient data
            logger.error("Not enough data or 'default_status' column missing. Training aborted.")
            return {"error": "Insufficient data for training."}

        X = df.drop(['default_status', 'loan_id', 'company_id'], axis=1)
        y = df['default_status']

        # Store feature names after one-hot encoding (requires fitting preprocessor first)
        # This is a bit tricky with ColumnTransformer; a common approach is to fit preprocessor separately
        # or extract after fitting the full pipeline.
        # For simplicity in PoC, we'll rely on the order defined in numerical/categorical_features.
        # A more robust way is to get feature names from the OneHotEncoder after fitting.

        # Ensure all defined features are present, add missing ones with NaNs or default values if necessary
        for feature_set in [self.numerical_features, self.categorical_features]:
            for col in feature_set:
                if col not in X.columns:
                    logger.warning(f"Feature '{col}' not found in prepared data. Adding it with NaNs.")
                    X[col] = pd.NA # Or some default like 0 for numerical, 'Unknown' for categorical


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None)

        if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
            logger.warning("Training or test set has only one class. Metrics might be misleading or fail.")
            # Handle this case, e.g., by skipping training or returning specific error
            if len(y_train.unique()) < 2:
                 logger.error("Training set has only one class. Cannot train RandomForestClassifier. Aborting.")
                 return {"error": "Training set has only one class."}

        # --- Conceptual Hyperparameter Tuning (GridSearchCV) ---
        # from sklearn.model_selection import GridSearchCV
        # param_grid = {
        #     'classifier__n_estimators': [50, 100, 200],
        #     'classifier__max_depth': [None, 10, 20],
        #     'classifier__min_samples_split': [2, 5, 10]
        # }
        # # Note: 'classifier__' prefix is used because RandomForestClassifier is a step in the Pipeline
        # grid_search = GridSearchCV(self.base_model, param_grid, cv=3, scoring='roc_auc', verbose=1, n_jobs=-1)
        # grid_search.fit(X_train, y_train)
        # logger.info(f"Best parameters found by GridSearchCV: {grid_search.best_params_}")
        # self.model = grid_search.best_estimator_ # Use the best model found
        # --- End Conceptual Hyperparameter Tuning ---
        # If not using GridSearchCV, fit the base_model directly:
        self.model = self.base_model.fit(X_train, y_train)

        # Extract feature names after fitting for interpretability (optional)
        # This is crucial for SHAP
        try:
            # Get feature names from the preprocessor step
            self._feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
            logger.info(f"Extracted feature names for SHAP: {self._feature_names}")
        except Exception as e:
            logger.error(f"Could not extract feature names using get_feature_names_out(): {e}")
            # Fallback: manually construct if the above fails (less robust)
            try:
                ohe_feature_names = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
                # Note: self.numerical_features should correspond to the columns passed to 'num' transformer
                # Ensure the order matches how ColumnTransformer concatenates them. Usually 'num' then 'cat'.
                self._feature_names = self.numerical_features + list(ohe_feature_names)
                logger.info(f"Fallback extracted feature names for SHAP: {self._feature_names}")
            except Exception as e_fallback:
                logger.error(f"Fallback for extracting feature names also failed: {e_fallback}")
                self._feature_names = X_train.columns.tolist() # Least robust fallback


        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)[:, 1] # Probability of class 1 (default)

        metrics = {
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "test_accuracy": accuracy_score(y_test, y_pred_test),
        }
        if len(y_test.unique()) > 1 : # ROC AUC requires at least two classes in y_true
             metrics["test_roc_auc"] = roc_auc_score(y_test, y_pred_proba_test)
        else:
            metrics["test_roc_auc"] = 0.0 # Or None, or some indicator

        logger.info(f"Training complete. Metrics: {metrics}")
        self.save_model()

        # Automate model registration
        if "error" not in metrics: # Register only if training was successful
            try:
                registry = ModelRegistry()
                model_params = self.model.named_steps['classifier'].get_params() if self.model else {}
                # Generate a simple timestamp-based version for PoC
                model_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

                registry.register_model(
                    model_name="PDModel",
                    model_version=model_version,
                    model_path=str(self.model_path),
                    metrics=metrics,
                    parameters=model_params,
                    tags={"stage": "training", "model_type": "RandomForestClassifier"}
                )
                logger.info(f"PDModel version {model_version} registered successfully.")
            except Exception as e:
                logger.error(f"Error during model registration for PDModel: {e}")

        return metrics

    def predict(self, new_data_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Makes PD predictions on new data (as a DataFrame).
        The input DataFrame should have the same raw features used during training.
        Returns a DataFrame with predictions and probabilities.
        """
        if self.model is None:
            logger.error("Model not trained or loaded. Call train() or load_model() first.")
            return None

        if new_data_df.empty:
            logger.warning("Input DataFrame for prediction is empty.")
            return pd.DataFrame(columns=['pd_prediction', 'pd_probability'])

        # Ensure all necessary columns are present, similar to training
        # This step is crucial and should match the feature engineering in _prepare_features
        # For PoC, assuming new_data_df is already structured like X in train (after _prepare_features, minus target)

        # Example: If new_data_df is raw (like from kb_service.get_company_profile + loan info)
        # It would need to go through a similar transformation as in _prepare_features
        # For this PoC, let's assume new_data_df is already somewhat processed
        # and matches the structure expected by the preprocessor.

        missing_cols = [col for col in self.numerical_features + self.categorical_features if col not in new_data_df.columns]
        if missing_cols:
            logger.error(f"Missing columns in prediction data: {missing_cols}. Cannot predict.")
            # Optionally, try to impute or add default values if appropriate for the application
            # For now, we'll fail.
            return None

        try:
            predictions = self.model.predict(new_data_df)
            probabilities = self.model.predict_proba(new_data_df)[:, 1] # Prob of default

            results_df = pd.DataFrame({
                'pd_prediction': predictions, # 0 or 1
                'pd_probability': probabilities # Probability of being 1
            })
            return results_df

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None


    def predict_for_loan(self, loan: Dict[str, Any], company: Dict[str, Any]) -> Optional[Tuple[int, float]]:
        """
        Predicts PD for a single loan and associated company data (as dicts).
        This is a convenience method.
        Returns a tuple (prediction (0/1), probability_of_default).
        """
        if self.model is None:
            logger.error("Model not trained or loaded.")
            return None

        # Feature Engineering - align with _prepare_features as much as possible
        # Time-based features
        loan_origination_date_obj = None
        if loan.get('origination_date'):
            try: loan_origination_date_obj = pd.to_datetime(loan['origination_date']).date()
            except: pass

        loan_maturity_date_obj = None
        if loan.get('maturity_date'):
            try: loan_maturity_date_obj = pd.to_datetime(loan['maturity_date']).date()
            except: pass

        loan_duration_days = -1
        if loan_maturity_date_obj and loan_origination_date_obj:
            loan_duration_days = (loan_maturity_date_obj - loan_origination_date_obj).days

        company_age_at_origination = -1
        company_founded_date_obj = None
        if company.get('founded_date'):
            try: company_founded_date_obj = pd.to_datetime(company['founded_date']).date()
            except: pass

        if company_founded_date_obj and loan_origination_date_obj:
            company_age_at_origination = (loan_origination_date_obj - company_founded_date_obj).days / 365.25

        # Financial Ratios - Will be NaN as we don't have FS here
        debt_to_equity_ratio = np.nan
        current_ratio = np.nan
        net_profit_margin = np.nan
        roe = np.nan

        # Interaction term
        loan_amount_val = loan.get('loan_amount', 0)
        interest_rate_val = loan.get('interest_rate_percentage', 0)
        loan_amount_x_interest_rate = loan_amount_val * interest_rate_val

        record = {
            'loan_amount_usd': loan_amount_val,
            'interest_rate_percentage': interest_rate_val,
            'collateral_type': str(loan.get('collateral_type', 'None')), # Categorical
            'industry_sector': str(company.get('industry_sector', 'Other')), # Categorical
            'loan_duration_days': loan_duration_days,
            'company_age_at_origination': company_age_at_origination,
            'debt_to_equity_ratio': debt_to_equity_ratio,
            'current_ratio': current_ratio,
            'net_profit_margin': net_profit_margin,
            'roe': roe,
            'loan_amount_x_interest_rate': loan_amount_x_interest_rate,
        }

        # Ensure all features defined in self.numerical_features and self.categorical_features are present
        # This loop is more for safety, explicit definition above is preferred.
        for col in self.numerical_features:
            if col not in record:
                record[col] = np.nan # Default for missing numerical
        for col in self.categorical_features:
            if col not in record:
                record[col] = 'Unknown' # Default for missing categorical

        df_record = pd.DataFrame([record])

        try:
            prediction = self.model.predict(df_record)[0]
            probability = self.model.predict_proba(df_record)[0, 1] # Prob of class 1
            return int(prediction), float(probability)
        except Exception as e:
            logger.error(f"Error predicting for single loan: {e}. Record: {df_record.to_dict()}")
            return None

    def save_model(self):
        """Saves the trained model to the specified path."""
        if self.model is not None:
            try:
                joblib.dump(self.model, self.model_path)
                logger.info(f"PD Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Error saving PD model: {e}")
        else:
            logger.warning("No model to save.")

    def load_model(self) -> bool:
        """Loads a pre-trained model from the specified path or falls back to registry."""
        model_loaded_successfully = False
        # Try loading from the specific model_path first (if it exists)
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"PD Model loaded from specified path: {self.model_path}")
                # Potentially load feature names if saved separately
                model_loaded_successfully = True
            except Exception as e:
                logger.error(f"Error loading PD model from {self.model_path}: {e}. Trying registry.")
                self.model = None # Ensure model is None if loading fails

        if not model_loaded_successfully:
            logger.info(f"Model file not found at {self.model_path} or failed to load. Attempting to load 'production' model from registry.")
            try:
                registry = ModelRegistry()
                prod_model_path_str = registry.get_production_model_path("PDModel")
                if prod_model_path_str:
                    prod_model_path = Path(prod_model_path_str)
                    if prod_model_path.exists():
                        self.model = joblib.load(prod_model_path)
                        self.model_path = prod_model_path # Update model_path to the one loaded
                        logger.info(f"PD Model (production) loaded from registry path: {self.model_path}")
                        model_loaded_successfully = True
                    else:
                        logger.warning(f"Production model path from registry does not exist: {prod_model_path}")
                else:
                    logger.warning("No production PDModel found in registry.")
            except Exception as e:
                logger.error(f"Error loading production PDModel from registry: {e}")
                self.model = None # Ensure model is None if registry loading fails

        if not model_loaded_successfully:
             logger.warning(f"PD Model could not be loaded from specified path or registry.")

        return model_loaded_successfully

    def get_feature_importance_shap(self, sample_instance_df: pd.DataFrame, num_explanations: int = 1) -> Optional[Dict[str, float]]:
        """
        Calculates SHAP feature importances for a sample instance (or multiple for summary).
        For RandomForestClassifier, this returns the mean absolute SHAP value for each feature.
        """
        if self.model is None:
            logger.error("Model is not trained or loaded. Cannot calculate SHAP values.")
            return None

        if not isinstance(self.model, Pipeline):
            logger.error("Model is not a scikit-learn Pipeline. SHAP explainability for this structure might not be supported.")
            return None

        if 'preprocessor' not in self.model.named_steps or 'classifier' not in self.model.named_steps:
            logger.error("Pipeline does not contain 'preprocessor' or 'classifier' steps.")
            return None

        classifier = self.model.named_steps['classifier']
        if not hasattr(classifier, 'predict_proba') or not (isinstance(classifier, RandomForestClassifier)):
            logger.warning(f"Classifier type {type(classifier)} may not be directly supported by shap.TreeExplainer or this SHAP implementation logic. Trying anyway.")

        if not hasattr(self, '_feature_names') or self._feature_names.size == 0:
            logger.error("Feature names are not available. Train model first or ensure _feature_names is populated.")
            return None

        try:
            preprocessor = self.model.named_steps['preprocessor']
            transformed_sample_instance = preprocessor.transform(sample_instance_df)

            if transformed_sample_instance.shape[1] != len(self._feature_names):
                logger.error(f"Mismatch in transformed feature count ({transformed_sample_instance.shape[1]}) and stored feature names count ({len(self._feature_names)}). SHAP values might be misaligned.")
                logger.error(f"Stored feature names: {self._feature_names}")
                try:
                    self._feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                    logger.info(f"Re-fetched feature names: {self._feature_names}")
                    if transformed_sample_instance.shape[1] != len(self._feature_names):
                         logger.error("Feature count mismatch persists even after re-fetching names. Aborting SHAP.")
                         return None
                except Exception as e_refetch:
                    logger.error(f"Failed to re-fetch feature names: {e_refetch}. Aborting SHAP.")
                    return None

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_sample_instance)

            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_for_positive_class = shap_values[1]
            else:
                shap_values_for_positive_class = shap_values

            if shap_values_for_positive_class.ndim == 1:
                mean_abs_shap = np.abs(shap_values_for_positive_class)
            else:
                mean_abs_shap = np.abs(shap_values_for_positive_class).mean(axis=0)

            feature_shap_dict = dict(zip(self._feature_names, mean_abs_shap))

            sorted_feature_shap = dict(sorted(feature_shap_dict.items(), key=lambda item: item[1], reverse=True))

            return sorted_feature_shap

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            logger.error(f"Sample instance columns: {sample_instance_df.columns.tolist()}")
            logger.error(f"Expected (numerical then categorical): {self.numerical_features} + {self.categorical_features}")

            return None

        try:
            preprocessor = self.model.named_steps['preprocessor']
            # Ensure sample_instance_df has the same columns as X used in train (before preprocessing)
            # These are the raw features defined at the start of _prepare_features
            # For this method, we expect a DataFrame with columns matching the *input* to _prepare_features,
            # so we must ensure it has the correct columns for the preprocessor.
            # The `sample_instance_df` should contain all columns listed in `self.numerical_features` and `self.categorical_features`
            # that are used by the preprocessor.

            # Transform the input data using the fitted preprocessor
            transformed_sample_instance = preprocessor.transform(sample_instance_df)

            # For some transformers (like SimpleImputer with add_indicator=True), the number of output features
            # might change. get_feature_names_out() on the preprocessor is the source of truth.
            if transformed_sample_instance.shape[1] != len(self._feature_names):
                logger.error(f"Mismatch in transformed feature count ({transformed_sample_instance.shape[1]}) and stored feature names count ({len(self._feature_names)}). SHAP values might be misaligned.")
                logger.error(f"Stored feature names: {self._feature_names}")
                # Attempt to re-fetch feature names if possible, this indicates an issue in train() or feature list management
                try:
                    self._feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
                    logger.info(f"Re-fetched feature names: {self._feature_names}")
                    if transformed_sample_instance.shape[1] != len(self._feature_names):
                         logger.error("Feature count mismatch persists even after re-fetching names. Aborting SHAP.")
                         return None
                except Exception as e_refetch:
                    logger.error(f"Failed to re-fetch feature names: {e_refetch}. Aborting SHAP.")
                    return None


            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_sample_instance)

            # For binary classification with RandomForest, shap_values is a list of two arrays (one for each class)
            # We use shap_values[1] for the positive class (default)
            # If multi-class, this would need adjustment.
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values_for_positive_class = shap_values[1]
            else: # Handles cases like single class SHAP values or other formats (though TreeExplainer usually gives list for RF)
                shap_values_for_positive_class = shap_values

            # Mean absolute SHAP value for each feature across all samples (if multiple passed)
            if shap_values_for_positive_class.ndim == 1: # Single instance explanation
                mean_abs_shap = np.abs(shap_values_for_positive_class)
            else: # Multiple instances
                mean_abs_shap = np.abs(shap_values_for_positive_class).mean(axis=0)

            feature_shap_dict = dict(zip(self._feature_names, mean_abs_shap))

            # Sort by importance (absolute SHAP value)
            sorted_feature_shap = dict(sorted(feature_shap_dict.items(), key=lambda item: item[1], reverse=True))

            return sorted_feature_shap

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            logger.error(f"Sample instance columns: {sample_instance_df.columns.tolist()}")
            logger.error(f"Expected (numerical then categorical): {self.numerical_features} + {self.categorical_features}")

            return None


if __name__ == "__main__":
    # This setup is for running the script directly for testing.
    # Ensure PYTHONPATH includes the project root.
    # Example: PYTHONPATH=. python src/risk_models/pd_model.py

    # Configure logging if not already set up by importing a core module
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing PDModel ---")
    kb = KnowledgeBaseService() # Uses sample data by default

    # Check if KB loaded data
    if not kb.get_all_companies() or not kb.get_all_loans():
        logger.error("KnowledgeBaseService could not load company or loan data. PD Model tests cannot proceed.")
    else:
        pd_model_instance = PDModel()

        # Train the model
        logger.info("Training PD model...")
        train_metrics = pd_model_instance.train(kb_service=kb)
        logger.info(f"PD Model training metrics: {train_metrics}")

        # After training, list models from registry (to show the new one)
        if "error" not in train_metrics:
            registry = ModelRegistry()
            logger.info("PD Models in registry after training:")
            pd_models_in_registry = registry.list_models("PDModel")
            for m in pd_models_in_registry:
                logger.info(f"  - Version: {m['model_version']}, Path: {m['model_path']}, Status: {m.get('status')}, Metrics: {m.get('metrics')}")

            # Promote the just trained model to "production" for testing fallback loading
            if pd_models_in_registry: # Check if any model was registered
                latest_version = pd_models_in_registry[0]['model_version'] # Assumes latest is first
                logger.info(f"Promoting PDModel version {latest_version} to 'production' for testing.")
                registry.update_model_status("PDModel", latest_version, "production")


        # Test fallback loading
        logger.info("--- Testing Fallback Model Loading for PDModel ---")
        # Create a new instance with a non-existent path to trigger fallback
        pd_model_for_fallback_test = PDModel(model_path=Path("models_store/non_existent_pd_model.joblib"))
        load_success_fallback = pd_model_for_fallback_test.load_model()
        logger.info(f"PDModel fallback loading attempt. Success: {load_success_fallback}")
        if load_success_fallback and pd_model_for_fallback_test.model is not None:
            logger.info(f"Fallback PDModel loaded from: {pd_model_for_fallback_test.model_path}")
        else:
            logger.warning("Fallback PDModel loading failed or no production model was found.")


        # Load the original model instance again to continue with its tests (if it was overwritten by fallback test)
        # Or simply use the existing pd_model_instance which should still hold the trained model
        # For clarity, let's assume pd_model_instance is still the primary one for subsequent tests.
        # If the fallback test modified a shared registry that affects original instance, behavior might change.
        # The registry is file-based, so pd_model_instance.load_model() would see changes if it reloads.

        logger.info("--- Resuming tests with originally trained/loaded PDModel instance ---")
        # Ensure the original instance has a model for subsequent tests
        if pd_model_instance.model is None: # If it wasn't loaded or trained successfully initially
            logger.info("Re-loading original PD model instance for subsequent tests...")
            load_success = pd_model_instance.load_model() # This will now try registry if primary path failed
            logger.info(f"Model loaded successfully for original instance: {load_success}")


        if pd_model_instance.model is not None:
            # Prepare some sample data for prediction (using one of the existing loans/companies)
            # Ensure kb_service is available and has data
            if not kb.get_all_loans():
                logger.error("No loans in KB for PDModel prediction test.")
                pass # or skip this part of test

            sample_loan_data = kb.get_all_loans()[0] # Take the first loan
            sample_company_data = kb.get_company_profile(sample_loan_data.company_id)

            if sample_loan_data and sample_company_data:
                logger.info(f"Predicting for single loan: {sample_loan_data.loan_id} of company {sample_company_data.company_id}")
                # Convert Pydantic models to dicts for predict_for_loan
                prediction_result = pd_model_instance.predict_for_loan(
                    sample_loan_data.model_dump(),
                    sample_company_data.model_dump()
                )
                if prediction_result:
                    pred_class, pred_proba = prediction_result
                    logger.info(f"Prediction: Class={pred_class}, Probability={pred_proba:.4f}")
                else:
                    logger.error("Failed to get prediction for single loan.")
            else:
                logger.warning("Could not fetch sample loan/company for single prediction test.")

            # Test batch prediction with a DataFrame
            # Create a DataFrame from a couple of samples for testing predict()
            # This needs to align with the new features. We'll use the first record from _prepare_features output
            # before it's split, to test batch predict and SHAP.

            full_feature_df = pd_model_instance._prepare_features(kb_service=kb) # Re-run to get the full df
            if not full_feature_df.empty:
                # Prepare a sample for predict() - needs columns expected by the preprocessor
                # These are the columns in X_train/X_test, which are listed in
                # model.numerical_features and model.categorical_features
                sample_for_batch_predict_df = full_feature_df.head(2).drop(['default_status', 'loan_id', 'company_id'], axis=1)

                # Ensure all required columns for the preprocessor are present
                missing_cols_predict = [col for col in pd_model_instance.numerical_features + pd_model_instance.categorical_features if col not in sample_for_batch_predict_df.columns]
                if missing_cols_predict:
                    logger.warning(f"Sample for batch predict is missing columns: {missing_cols_predict}. Adding with NaN/Unknown.")
                    for col in missing_cols_predict:
                        sample_for_batch_predict_df[col] = np.nan if col in pd_model_instance.numerical_features else 'Unknown'

                logger.info(f"Predicting for batch data (first {len(sample_for_batch_predict_df)} loans):")
                logger.info(f"Batch predict input columns: {sample_for_batch_predict_df.columns.tolist()}")
                batch_predictions = pd_model_instance.predict(sample_for_batch_predict_df)
                if batch_predictions is not None:
                    logger.info(f"Batch predictions:\n{batch_predictions}")
                else:
                    logger.error("Failed to get batch predictions.")

                # Test SHAP Feature Importance
                # SHAP also expects a DataFrame with columns that the preprocessor was trained on.
                # Use the same sample_for_batch_predict_df or a single instance from it.
                logger.info("--- Testing SHAP Feature Importance ---")
                # Take the first instance from the sample batch
                single_instance_for_shap = sample_for_batch_predict_df.head(1)
                logger.info(f"Columns for SHAP sample: {single_instance_for_shap.columns.tolist()}")

                shap_importances = pd_model_instance.get_feature_importance_shap(single_instance_for_shap)
                if shap_importances:
                    logger.info("SHAP Feature Importances (Mean Absolute SHAP for class 1):")
                    for feature, importance in shap_importances.items():
                        logger.info(f"  {feature}: {importance:.4f}")
                else:
                    logger.warning("Could not retrieve SHAP feature importances.")
            else:
                logger.warning("Full feature DataFrame is empty, skipping batch prediction and SHAP tests.")
        else:
            logger.error("PD Model could not be loaded. Prediction and SHAP tests skipped.")
