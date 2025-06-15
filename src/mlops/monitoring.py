import logging
from typing import Dict, Any, List, Optional
import datetime
import pandas as pd
import numpy as np # Added for PoC drift check
import json # Added for log_prediction

# from evidently.report import Report # Example for advanced monitoring
# from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset # Example

logger = logging.getLogger(__name__)

class ModelMonitor:
    """
    Conceptual placeholder for Model Monitoring.
    In a real MLOps pipeline, this would involve logging predictions, checking for data drift,
    concept drift, and model performance degradation using tools like Evidently AI, Prometheus, Grafana, etc.
    """
    def __init__(self, monitoring_log_path: Optional[str] = None):
        self.monitoring_log_path = monitoring_log_path or "model_monitoring_log.jsonl"
        # In a real system, this would likely log to a database or a dedicated logging service.
        logger.info(f"ModelMonitor initialized. Logging predictions conceptually to: {self.monitoring_log_path}")

    def log_prediction(self,
                       model_name: str,
                       model_version: str,
                       request_id: str,
                       features: Dict[str, Any],
                       prediction: Any,
                       probability: Optional[float] = None,
                       timestamp: Optional[str] = None):
        """
        Logs input features and output predictions for a given model.
        This data can be used for drift detection and performance analysis.
        """
        log_entry = {
            "timestamp": timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "request_id": request_id,
            "features": features,
            "prediction": prediction,
            "probability": probability,
        }

        try:
            with open(self.monitoring_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            # logger.debug(f"Logged prediction for model {model_name} v{model_version}, request {request_id}")
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")


    def check_data_drift(self,
                         model_name: str,
                         reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         column_mapping: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        """
        Conceptual check for data drift between a reference dataset (e.g., training data)
        and current production data.

        A real implementation would use statistical tests (e.g., KS test, PSI).
        Tools like Evidently AI can automate this.
        """
        logger.info(f"Conceptually checking data drift for model: {model_name}")
        if reference_data.empty or current_data.empty:
            logger.warning("Reference or current data is empty. Cannot perform drift check.")
            return None

        # Example: Using Evidently AI (conceptual, not run in PoC)
        # if 'evidently' in sys.modules:
        #     data_drift_report = Report(metrics=[DataDriftPreset(num_stattest='ks', cat_stattest='psi')])
        #     data_drift_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)
        #     drift_results = data_drift_report.as_dict()
        #     logger.info("Evidently AI data drift report (conceptual):")
        #     # logger.info(json.dumps(drift_results, indent=2)) # This would be very verbose
        #     return drift_results['metrics'][0]['result'] # Example path to drift status
        # else:
        #    logger.warning("Evidently AI not installed. Skipping advanced drift check.")

        # Simple PoC drift check: Compare means and std deviations for numerical features
        drift_details = {}
        numerical_cols = reference_data.select_dtypes(include=np.number).columns
        categorical_cols = reference_data.select_dtypes(include='object').columns # Assuming object type for categorical

        for col in numerical_cols:
            if col in current_data.columns and col in reference_data.columns:
                ref_mean = reference_data[col].mean()
                cur_mean = current_data[col].mean()
                ref_std = reference_data[col].std()
                cur_std = current_data[col].std()

                # Handle cases where std might be zero
                mean_drift = abs(ref_mean - cur_mean) / (ref_std + 1e-9)
                std_drift = abs(ref_std - cur_std) / (ref_std + 1e-9)

                drift_details[f"{col}_numerical"] = { # Added suffix to avoid potential name collision
                    "type": "numerical",
                    "reference_mean": ref_mean, "current_mean": cur_mean, "mean_drift_score": mean_drift,
                    "reference_std": ref_std, "current_std": cur_std, "std_drift_score": std_drift,
                    "is_drifted": mean_drift > 0.2 or std_drift > 0.2 # Example threshold
                }
            else:
                 drift_details[f"{col}_numerical"] = {"type": "numerical", "status": "missing_in_current_or_ref", "is_drifted": True}


        # Enhanced check for categorical features
        for col in categorical_cols:
            if col in current_data.columns and col in reference_data.columns:
                ref_counts = reference_data[col].value_counts(normalize=True)
                cur_counts = current_data[col].value_counts(normalize=True)

                cat_drift_info = {"type": "categorical", "is_drifted": False, "details": {}}

                # Check for new categories in current data
                new_categories = set(cur_counts.index) - set(ref_counts.index)
                if new_categories:
                    cat_drift_info["is_drifted"] = True
                    cat_drift_info["details"]["new_categories"] = list(new_categories)

                # Check for missing categories or significant proportion changes
                for category, ref_prop in ref_counts.items():
                    cur_prop = cur_counts.get(category, 0.0)
                    prop_diff = abs(ref_prop - cur_prop)
                    cat_drift_info["details"][f"{category}_prop_diff"] = prop_diff
                    if prop_diff > 0.2: # Example threshold for significant proportion change
                        cat_drift_info["is_drifted"] = True

                drift_details[f"{col}_categorical"] = cat_drift_info
            else:
                drift_details[f"{col}_categorical"] = {"type": "categorical", "status": "missing_in_current_or_ref", "is_drifted": True}


        overall_drift_detected = any(details.get("is_drifted", False) for details in drift_details.values())
        logger.info(f"PoC Data Drift Check for {model_name}: Overall drift detected = {overall_drift_detected}")
        # Load the reference data for PDModel if model_name matches
        # This part seems misplaced. Reference data should be an argument or loaded at start of function.
        # Assuming reference_data is already passed as an argument.
        # if model_name == "PDModel":
        #     try:
        #         ref_data_path = "data/reference_pd_features.csv" # Path to the reference data
        #         reference_data_pd_model = pd.read_csv(ref_data_path)
        #         # Now compare current_data with reference_data_pd_model
        #         # ... (rest of the comparison logic using reference_data_pd_model)
        #     except FileNotFoundError:
        #         logger.error(f"Reference data file for PDModel not found at {ref_data_path}. Cannot perform specific PD drift check.")
        #         return {"error": "Reference data for PDModel not found."}


        return {"poc_drift_details": drift_details, "overall_drift_detected": overall_drift_detected}

    def _simulate_new_ground_truth(self,
                                   logged_predictions: List[Dict[str, Any]],
                                   model_name_filter: str,
                                   target_column_name: str = 'default_status',
                                   known_good_probability_threshold: float = 0.1,
                                   known_bad_probability_threshold: float = 0.8,
                                   corruption_rate: float = 0.05) -> pd.DataFrame:
        """
        Simulates new ground truth for a list of logged predictions.
        Focuses on PDModel ('default_status') for this implementation.
        """
        if not logged_predictions:
            return pd.DataFrame()

        # Filter for the relevant model and prepare data
        data_for_df = []
        for log_entry in logged_predictions:
            if log_entry.get('model_name') == model_name_filter:
                # Basic feature set for context, can be expanded if needed by specific models
                record = {
                    'prediction': log_entry.get('prediction'),
                    'probability': log_entry.get('probability'),
                    # Include some key features if they are simple types and available
                    # For complex features, this might need adjustment or rely on request_id to fetch full features
                }
                # Example: Add a few simple features if they exist in the log
                if isinstance(log_entry.get('features'), dict):
                    for f_key in ['loan_amount_usd', 'collateral_type', 'industry_sector']: # Example features
                         if f_key in log_entry['features']:
                            record[f_key] = log_entry['features'][f_key]
                data_for_df.append(record)

        if not data_for_df:
            logger.warning(f"No logged predictions found for model {model_name_filter} to simulate ground truth.")
            return pd.DataFrame()

        df = pd.DataFrame(data_for_df)

        actual_target_col_name = f"actual_{target_column_name}"
        df[actual_target_col_name] = 0 # Default to 0 (no default)

        if model_name_filter == "PDModel":
            if 'probability' not in df.columns or 'prediction' not in df.columns:
                logger.error("PDModel logs missing 'probability' or 'prediction' for ground truth simulation.")
                return df # Or empty df

            # Initial assignment based on probability thresholds
            df.loc[df['probability'] < known_good_probability_threshold, actual_target_col_name] = 0
            df.loc[df['probability'] > known_bad_probability_threshold, actual_target_col_name] = 1

            # For intermediate probabilities, use original prediction (or could be random based on prob)
            intermediate_mask = (df['probability'] >= known_good_probability_threshold) & \
                                (df['probability'] <= known_bad_probability_threshold)
            df.loc[intermediate_mask, actual_target_col_name] = df.loc[intermediate_mask, 'prediction']

            # Apply corruption_rate
            num_corruptions = int(len(df) * corruption_rate)
            if num_corruptions > 0 and not df.empty:
                corruption_indices = np.random.choice(df.index, num_corruptions, replace=False)
                df.loc[corruption_indices, actual_target_col_name] = 1 - df.loc[corruption_indices, actual_target_col_name] # Flip bit

        elif model_name_filter == "LGDModel":
            # Conceptual: LGD ground truth simulation would be different.
            # For now, we'll just return the df without specific LGD ground truth.
            # Could involve taking predicted LGD (from 'prediction' field, assuming it's LGD),
            # converting to recovery_rate, adding noise/shift, and storing as 'actual_recovery_rate'.
            # Example: df['actual_recovery_rate'] = np.clip(1.0 - df['prediction'] + np.random.normal(0, 0.1, len(df)), 0.05, 0.95)
            logger.info(f"Ground truth simulation for LGDModel ('{target_column_name}') is conceptual. No actuals generated beyond initial df structure.")
            if target_column_name not in df.columns: # Ensure the column exists if it was expected
                df[actual_target_col_name] = np.nan


        return df


    def check_model_performance_degradation(self,
                                           model_name: str,
                                           logged_predictions_path: str, # Path to JSONL file
                                           reference_metrics: Dict[str, float], # e.g. {"accuracy": 0.85, "roc_auc": 0.90}
                                           metric_to_check: str = "accuracy", # or "roc_auc", "mse" etc.
                                           threshold_percentage_drop: float = 10.0,
                                           target_column_name: str = 'default_status' # For PDModel
                                           ) -> Dict[str, Any]:
        """
        Checks for model performance degradation using logged predictions and simulated ground truth.
        """
        logger.info(f"Checking performance degradation for model: {model_name} using log file: {logged_predictions_path}")

        try:
            logged_predictions = []
            with open(logged_predictions_path, 'r') as f:
                for line in f:
                    logged_predictions.append(json.loads(line))
        except FileNotFoundError:
            logger.error(f"Prediction log file not found: {logged_predictions_path}")
            return {"status": "error", "message": "Prediction log file not found."}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from prediction log file: {logged_predictions_path}")
            return {"status": "error", "message": "Error decoding prediction log."}

        if not logged_predictions:
            logger.warning("Prediction log is empty. Cannot check performance.")
            return {"status": "no_data", "message": "Prediction log empty."}

        # Simulate new ground truth
        # Pass model_name to _simulate_new_ground_truth for model-specific logic
        simulated_data_df = self._simulate_new_ground_truth(logged_predictions, model_name_filter=model_name, target_column_name=target_column_name)

        if simulated_data_df.empty:
            logger.warning(f"No data generated from simulation for model {model_name}. Cannot check performance.")
            return {"status": "no_simulated_data", "message": "Simulated data is empty."}

        actual_target_col = f"actual_{target_column_name}"
        if actual_target_col not in simulated_data_df.columns:
             logger.error(f"Simulated actual target column '{actual_target_col}' not found in DataFrame.")
             return {"status": "error", "message": f"Simulated actual target column missing."}


        current_metric_value = 0.0
        # Ensure predictions and actuals are available for metric calculation
        if 'prediction' not in simulated_data_df.columns and metric_to_check == "accuracy": # or other metrics needing 'prediction'
            logger.error(f"'prediction' column not found in simulated data for model {model_name} to calculate {metric_to_check}.")
            return {"status": "error", "message": f"'prediction' column missing for {metric_to_check}."}
        if 'probability' not in simulated_data_df.columns and metric_to_check == "roc_auc":
            logger.error(f"'probability' column not found in simulated data for model {model_name} to calculate roc_auc.")
            return {"status": "error", "message": "'probability' column missing for roc_auc."}


        from sklearn.metrics import accuracy_score, roc_auc_score # Import here to keep it local if only used here

        try:
            if metric_to_check == "accuracy":
                if pd.isna(simulated_data_df['prediction']).any() or pd.isna(simulated_data_df[actual_target_col]).any():
                    logger.warning("NaNs found in prediction or actuals for accuracy calculation. Dropping NaNs.")
                    temp_df = simulated_data_df[['prediction', actual_target_col]].dropna()
                    if temp_df.empty: raise ValueError("No valid data after dropping NaNs for accuracy.")
                    current_metric_value = accuracy_score(temp_df[actual_target_col], temp_df['prediction'])
                else:
                    current_metric_value = accuracy_score(simulated_data_df[actual_target_col], simulated_data_df['prediction'])
            elif metric_to_check == "roc_auc":
                if pd.isna(simulated_data_df['probability']).any() or pd.isna(simulated_data_df[actual_target_col]).any():
                    logger.warning("NaNs found in probability or actuals for ROC AUC calculation. Dropping NaNs.")
                    temp_df = simulated_data_df[['probability', actual_target_col]].dropna()
                    if temp_df.empty: raise ValueError("No valid data after dropping NaNs for ROC AUC.")
                    if len(temp_df[actual_target_col].unique()) < 2 :
                        logger.warning(f"Only one class present in y_true after NaNs dropped for ROC AUC. Setting to 0.0 for {model_name}.")
                        current_metric_value = 0.0
                    else:
                        current_metric_value = roc_auc_score(temp_df[actual_target_col], temp_df['probability'])
                elif len(simulated_data_df[actual_target_col].unique()) < 2 :
                     logger.warning(f"Only one class present in y_true for ROC AUC calculation. Setting to 0.0 for {model_name}.")
                     current_metric_value = 0.0
                else:
                    current_metric_value = roc_auc_score(simulated_data_df[actual_target_col], simulated_data_df['probability'])

            # Add other metrics like MSE for LGDModel if needed
            # elif metric_to_check == "mse" and model_name == "LGDModel":
            #     # Assuming 'prediction' for LGDModel is the predicted LGD, and actual_target_col is 'actual_lgd'
            #     # current_metric_value = mean_squared_error(simulated_data_df[actual_target_col], simulated_data_df['prediction'])
            #     pass # Placeholder for LGD metric
            else:
                logger.warning(f"Metric '{metric_to_check}' not implemented for model {model_name} in this PoC.")
                return {"status": "error", "message": f"Metric '{metric_to_check}' not implemented."}
        except ValueError as ve:
            logger.error(f"ValueError during metric calculation for {model_name} ({metric_to_check}): {ve}")
            return {"status": "error", "message": f"ValueError during metric calculation: {ve}"}


        reference_value = reference_metrics.get(metric_to_check)
        if reference_value is None:
            logger.error(f"Reference metric '{metric_to_check}' not found for model {model_name}.")
            return {"status": "error", "message": "Reference metric not found."}

        degradation = False
        percentage_change = 0.0 # Default to 0 if reference_value is 0 or current_metric_value is not comparable
        if reference_value != 0: # Avoid division by zero
            # For metrics like MSE, lower is better, so degradation is an increase.
            if metric_to_check in ["mse", "rmse"]: # Add other "lower is better" metrics here
                percentage_change = ((current_metric_value - reference_value) / reference_value) * 100
                if percentage_change > threshold_percentage_drop: # Degradation if current is WORSE (higher) by threshold
                    degradation = True
            else: # For metrics like accuracy, roc_auc, higher is better
                percentage_change = ((reference_value - current_metric_value) / reference_value) * 100
                if percentage_change > threshold_percentage_drop: # Degradation if current is WORSE (lower) by threshold
                    degradation = True
        elif current_metric_value != reference_value : # If reference is 0, any non-zero current value (if worse) is degradation
             # This logic needs to be metric specific. e.g. if ref_auc = 0 and cur_auc = 0.5, that's an improvement.
             # If ref_mse = 0 and cur_mse = 0.1, that's degradation.
             # For simplicity, if ref is 0, and current is also 0, it's not degradation. Otherwise, it's hard to define percentage.
             logger.warning(f"Reference value for {metric_to_check} is 0. Percentage change calculation might be misleading.")
             if metric_to_check in ["mse", "rmse"] and current_metric_value > 0 : degradation = True # Any error is worse than 0 error
             # For 'higher is better' metrics, if ref is 0, any positive current is improvement, negative or zero is same/worse.
             # This part is tricky, for now, if ref is 0, we assume no degradation if current is also 0, otherwise it depends.


        logger.info(f"Performance check for {model_name} on metric '{metric_to_check}': "
                    f"Reference={reference_value:.4f}, Current={current_metric_value:.4f}, "
                    f"Degradation Detected={degradation} ({percentage_change:.2f}% change)")

        return {
            "metric_checked": metric_to_check,
            "reference_value": reference_value,
            "current_value": current_metric_value,
            "percentage_change": percentage_change,
            "degradation_detected": degradation
        }

if __name__ == "__main__":
    import uuid
    # import numpy as np # Already imported at the top
    from pathlib import Path # Added for file cleanup

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing ModelMonitor ---")

    # Define paths for dummy files
    dummy_reference_features_path = Path("data/reference_pd_features.csv") # Used in previous step
    dummy_prediction_log_path = Path("test_model_monitor_log.jsonl")

    monitor = ModelMonitor(monitoring_log_path=str(dummy_prediction_log_path))

    # --- Test Data Drift ---
    logger.info("\n--- Testing Data Drift ---")
    if not dummy_reference_features_path.exists():
        logger.error(f"Dummy reference file {dummy_reference_features_path} not found. Create it first (as in previous step). Skipping drift test.")
    else:
        reference_df_loaded = pd.read_csv(dummy_reference_features_path)
        logger.info(f"Loaded reference data for drift check from: {dummy_reference_features_path} with {len(reference_df_loaded)} rows.")

        # Create sample current data for drift testing
        # Scenario 1: No significant drift
        current_data_no_drift = reference_df_loaded.sample(n=3, random_state=42).copy() if len(reference_df_loaded) >=3 else reference_df_loaded.copy()
        # Add some minor variations if desired, or just use a sample
        current_data_no_drift['loan_amount_usd'] = current_data_no_drift['loan_amount_usd'] * 1.05

        logger.info("\nChecking data drift (expecting low/no drift):")
        drift_results_no = monitor.check_data_drift("PDModel", reference_df_loaded, current_data_no_drift)
        if drift_results_no:
            logger.info(f"Overall drift detected (low/no drift case): {drift_results_no.get('overall_drift_detected')}")
            # logger.info(f"Details: {json.dumps(drift_results_no.get('poc_drift_details'), indent=2, default=str)}")


        # Scenario 2: With significant drift
        current_data_with_drift = pd.DataFrame({
            'loan_amount_usd': [10000, 15000, 20000000], # Shifted mean, increased std
            'interest_rate_percentage': [10.0, 12.5, 15.0], # Shifted
            'loan_duration_days': [360, 720, 100], # Different scale
            'company_age_at_origination': [0.5, 1.0, 0.2], # Shifted
            'debt_to_equity_ratio': [3.0, 4.0, 5.0], # Shifted
            'current_ratio': [0.5, 0.4, 0.3], # Shifted
            'net_profit_margin': [-0.5, -0.2, -0.1], # Shifted
            'roe': [-0.8, -0.5, -0.3], # Shifted
            'loan_amount_x_interest_rate': [100000, 187500, 300000000], # Shifted
            'industry_sector': ['NewSector', 'Technology', 'NewSector'], # New category, changed distribution
            'collateral_type': ['None', 'Exotic', 'Real Estate'] # New category
        })
        logger.info("\nChecking data drift (expecting drift):")
        drift_results_yes = monitor.check_data_drift("PDModel", reference_df_loaded, current_data_with_drift)
        if drift_results_yes:
            logger.info(f"Overall drift detected (drift case): {drift_results_yes.get('overall_drift_detected')}")
            # logger.info(f"Details: {json.dumps(drift_results_yes.get('poc_drift_details'), indent=2, default=str)}")


    # --- Test Performance Degradation ---
    logger.info("\n--- Testing Performance Degradation ---")
    # Create dummy prediction log file
    sample_log_entries = [
        {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_name": "PDModel", "model_version": "1.2.3", "request_id": "uuid1", "features": {"loan_amount_usd": 1000}, "prediction": 0, "probability": 0.05},
        {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_name": "PDModel", "model_version": "1.2.3", "request_id": "uuid2", "features": {"loan_amount_usd": 2000}, "prediction": 0, "probability": 0.15},
        {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_name": "PDModel", "model_version": "1.2.3", "request_id": "uuid3", "features": {"loan_amount_usd": 3000}, "prediction": 1, "probability": 0.85},
        {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_name": "PDModel", "model_version": "1.2.3", "request_id": "uuid4", "features": {"loan_amount_usd": 4000}, "prediction": 1, "probability": 0.95},
        {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_name": "PDModel", "model_version": "1.2.3", "request_id": "uuid5", "features": {"loan_amount_usd": 5000}, "prediction": 0, "probability": 0.45} # Intermediate
    ]
    with open(dummy_prediction_log_path, 'w') as f:
        for entry in sample_log_entries:
            f.write(json.dumps(entry) + '\n')
    logger.info(f"Created dummy prediction log: {dummy_prediction_log_path}")

    reference_pd_metrics = {"accuracy": 0.80, "roc_auc": 0.90} # Example reference metrics

    logger.info("\nChecking PDModel performance degradation (ROC AUC):")
    perf_check_roc_auc = monitor.check_model_performance_degradation(
        model_name="PDModel",
        logged_predictions_path=str(dummy_prediction_log_path),
        reference_metrics=reference_pd_metrics,
        metric_to_check="roc_auc",
        target_column_name='default_status', # As used in _simulate_new_ground_truth
        corruption_rate=0.0 # No corruption for this test to see baseline from thresholds
    )
    logger.info(f"Performance degradation check (ROC AUC) results: {perf_check_roc_auc}")

    logger.info("\nChecking PDModel performance degradation (Accuracy, with corruption):")
    perf_check_accuracy = monitor.check_model_performance_degradation(
        model_name="PDModel",
        logged_predictions_path=str(dummy_prediction_log_path),
        reference_metrics=reference_pd_metrics,
        metric_to_check="accuracy",
        target_column_name='default_status',
        corruption_rate=0.2 # Simulate 20% label corruption
    )
    logger.info(f"Performance degradation check (Accuracy) results: {perf_check_accuracy}")


    # Clean up dummy files
    if dummy_prediction_log_path.exists():
        dummy_prediction_log_path.unlink()
        logger.info(f"\nCleaned up test prediction log: {dummy_prediction_log_path}")

    # Note: dummy_reference_features_path is not cleaned up here as it's created in a previous step
    # and might be useful if tests are run iteratively. It could be added to cleanup if desired.
