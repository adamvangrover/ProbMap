import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Any, Optional, Dict
from pathlib import Path

from src.core.config import settings # For model artifact path
from src.data_management.knowledge_base import KnowledgeBaseService # To get data for training/prediction

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
        self.numerical_features = ['loan_amount_usd', 'interest_rate_percentage', 'company_revenue_usd_millions', 'company_age_years']
        self.categorical_features = ['industry_sector', 'collateral_type'] # company_country_iso_code could be another

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), self.numerical_features),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), self.categorical_features)
            ],
            remainder='drop' # or 'passthrough'
        )
        self.base_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

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

            # Basic feature engineering
            company_age = -1 # Default if no founded_date
            if company.founded_date:
                company_age = (pd.Timestamp('today').date() - company.founded_date).days / 365.25

            record = {
                'loan_id': loan.loan_id,
                'company_id': loan.company_id,
                'loan_amount_usd': loan.loan_amount,
                'interest_rate_percentage': loan.interest_rate_percentage,
                'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None',
                'industry_sector': company.industry_sector.value if company.industry_sector else 'Other',
                'company_revenue_usd_millions': company.revenue_usd_millions if company.revenue_usd_millions is not None else 0,
                'company_age_years': company_age,
                'default_status': int(loan.default_status) # Target variable
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
                 logger.error("Training set has only one class. Cannot train Logistic Regression. Aborting.")
                 return {"error": "Training set has only one class."}


        self.model = self.base_model.fit(X_train, y_train)

        # Extract feature names after fitting for interpretability (optional)
        try:
            ohe_feature_names = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self._feature_names = self.numerical_features + list(ohe_feature_names)
        except Exception as e:
            logger.warning(f"Could not extract feature names after OHE: {e}")
            self._feature_names = X_train.columns.tolist() # Fallback, might not be fully accurate with OHE

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

        # Construct a DataFrame for the single prediction
        company_age = -1
        if company.get('founded_date'):
            # Ensure founded_date is a date object if string
            founded_dt = company['founded_date']
            if isinstance(founded_dt, str):
                try:
                    founded_dt = pd.to_datetime(founded_dt).date()
                except ValueError:
                    logger.warning(f"Could not parse founded_date string: {founded_dt}")
                    founded_dt = None
            if founded_dt:
                 company_age = (pd.Timestamp('today').date() - founded_dt).days / 365.25

        record = {
            'loan_amount_usd': loan.get('loan_amount'), # Assuming 'loan_amount' from LoanAgreement
            'interest_rate_percentage': loan.get('interest_rate_percentage'),
            'collateral_type': str(loan.get('collateral_type', 'None')),
            'industry_sector': str(company.get('industry_sector', 'Other')),
            'company_revenue_usd_millions': company.get('revenue_usd_millions', 0),
            'company_age_years': company_age,
        }
        # Need to ensure all features defined in self.numerical_features and self.categorical_features are present
        # Add any missing features with default values if they were part of training
        for col in self.numerical_features + self.categorical_features:
            if col not in record:
                # A simple default strategy, might need refinement
                record[col] = 0 if col in self.numerical_features else 'Unknown'


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
        """Loads a pre-trained model from the specified path."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"PD Model loaded from {self.model_path}")
                # Potentially load feature names if saved separately
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}. Model not loaded.")
                return False
        except Exception as e:
            logger.error(f"Error loading PD model: {e}")
            self.model = None # Ensure model is None if loading fails
            return False

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

        # Load the model (even though it's already in memory, to test loading)
        logger.info("Loading PD model...")
        load_success = pd_model_instance.load_model()
        logger.info(f"Model loaded successfully: {load_success}")

        if load_success and pd_model_instance.model is not None:
            # Prepare some sample data for prediction (using one of the existing loans/companies)
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
            test_records_for_predict = []
            loans_for_predict = kb.get_all_loans()[:2] # Get first two loans
            for l_data in loans_for_predict:
                c_data = kb.get_company_profile(l_data.company_id)
                if c_data:
                    company_age_val = -1
                    if c_data.founded_date:
                        company_age_val = (pd.Timestamp('today').date() - c_data.founded_date).days / 365.25

                    test_records_for_predict.append({
                        'loan_amount_usd': l_data.loan_amount,
                        'interest_rate_percentage': l_data.interest_rate_percentage,
                        'collateral_type': l_data.collateral_type.value if l_data.collateral_type else 'None',
                        'industry_sector': c_data.industry_sector.value if c_data.industry_sector else 'Other',
                        'company_revenue_usd_millions': c_data.revenue_usd_millions if c_data.revenue_usd_millions is not None else 0,
                        'company_age_years': company_age_val
                    })

            if test_records_for_predict:
                test_df_for_predict = pd.DataFrame(test_records_for_predict)
                logger.info(f"Predicting for batch data (first {len(test_df_for_predict)} loans):")
                batch_predictions = pd_model_instance.predict(test_df_for_predict)
                if batch_predictions is not None:
                    logger.info(f"Batch predictions:\n{batch_predictions}")
                else:
                    logger.error("Failed to get batch predictions.")
            else:
                logger.warning("Not enough data to form a batch for prediction test.")

        else:
            logger.error("PD Model could not be loaded. Prediction tests skipped.")
