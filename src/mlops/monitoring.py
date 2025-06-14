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
        for col in reference_data.select_dtypes(include=np.number).columns:
            if col in current_data.columns:
                ref_mean = reference_data[col].mean()
                cur_mean = current_data[col].mean()
                ref_std = reference_data[col].std()
                cur_std = current_data[col].std()

                mean_drift = abs(ref_mean - cur_mean) / (ref_std + 1e-6) # Relative to ref_std
                std_drift = abs(ref_std - cur_std) / (ref_std + 1e-6)

                drift_details[col] = {
                    "reference_mean": ref_mean, "current_mean": cur_mean, "mean_drift_score": mean_drift,
                    "reference_std": ref_std, "current_std": cur_std, "std_drift_score": std_drift,
                    "is_drifted": mean_drift > 0.2 or std_drift > 0.2 # Example threshold
                }

        overall_drift_detected = any(details.get("is_drifted", False) for details in drift_details.values())
        logger.info(f"PoC Data Drift Check for {model_name}: Overall drift detected = {overall_drift_detected}")
        return {"poc_drift_details": drift_details, "overall_drift_detected": overall_drift_detected}


    def check_model_performance_degradation(self, model_name: str, validation_data_with_actuals: pd.DataFrame,
                                           model_predict_function, # Function that takes df and returns predictions
                                           reference_metrics: Dict[str, float], # e.g. {"accuracy": 0.85}
                                           metric_to_check: str = "accuracy", # or "roc_auc", "mse" etc.
                                           threshold_percentage_drop: float = 10.0
                                           ) -> Dict[str, Any]:
        """
        Conceptual check for model performance degradation using new labeled data.
        """
        logger.info(f"Conceptually checking performance degradation for model: {model_name}")
        if validation_data_with_actuals.empty:
            logger.warning("Validation data is empty. Cannot check performance.")
            return {"status": "no_data", "message": "Validation data empty."}

        # This assumes validation_data_with_actuals has features and a 'target' column
        # X_val = validation_data_with_actuals.drop('target', axis=1)
        # y_val_actual = validation_data_with_actuals['target']
        # y_val_pred = model_predict_function(X_val)

        # For PoC, let's simulate this.
        # In reality, you'd calculate the chosen metric (e.g. accuracy) on new predictions.
        # current_metric_value = accuracy_score(y_val_actual, y_val_pred)

        # Simulated current metric for PoC
        current_metric_value = reference_metrics.get(metric_to_check, 0.0) * (1 - np.random.uniform(0.0, 0.3)) # Simulate some degradation

        reference_value = reference_metrics.get(metric_to_check)
        if reference_value is None:
            logger.error(f"Reference metric '{metric_to_check}' not found for model {model_name}.")
            return {"status": "error", "message": "Reference metric not found."}

        degradation = False
        percentage_change = 0
        if reference_value != 0: # Avoid division by zero
            percentage_change = ((reference_value - current_metric_value) / reference_value) * 100
            if percentage_change > threshold_percentage_drop:
                degradation = True

        logger.info(f"Performance check for {model_name} on metric '{metric_to_check}': "
                    f"Reference={reference_value:.4f}, Current={current_metric_value:.4f}, "
                    f"Degradation Detected={degradation} ({percentage_change:.2f}% drop)")

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
    monitor = ModelMonitor(monitoring_log_path="test_model_monitor_log.jsonl") # Use test log

    # Test logging predictions
    monitor.log_prediction("PDModel", "1.1.0", str(uuid.uuid4()), {"feature1": 10, "feature2": "A"}, 0, 0.123)
    monitor.log_prediction("LGDModel", "1.0.0", str(uuid.uuid4()), {"collateral": "Real Estate"}, 0.25)
    logger.info(f"Test predictions logged to {monitor.monitoring_log_path}")

    # Test data drift (PoC version)
    reference_df = pd.DataFrame({
        'age': np.random.normal(40, 10, 1000),
        'income': np.random.gamma(2, scale=10000, size=1000),
        'city': np.random.choice(['A', 'B', 'C'], 1000)
    })
    current_df_no_drift = reference_df.sample(500).copy() # No drift
    current_df_with_drift = pd.DataFrame({ # Drifted data
        'age': np.random.normal(45, 12, 500), # Mean and std shifted
        'income': np.random.gamma(2.5, scale=12000, size=500), # Distribution shifted
        'city': np.random.choice(['A', 'B', 'D'], 500) # Category D introduced
    })

    logger.info("\nChecking data drift (no actual drift expected):")
    drift_results_no = monitor.check_data_drift("TestModel", reference_df, current_df_no_drift)
    if drift_results_no: logger.info(f"Overall drift detected (no drift case): {drift_results_no.get('overall_drift_detected')}")

    logger.info("\nChecking data drift (drift expected):")
    drift_results_yes = monitor.check_data_drift("TestModel", reference_df, current_df_with_drift)
    if drift_results_yes:
        logger.info(f"Overall drift detected (drift case): {drift_results_yes.get('overall_drift_detected')}")
        # logger.info("Drift details (drift case):")
        # for col, details in drift_results_yes.get('poc_drift_details', {}).items():
        #     logger.info(f"  {col}: Mean Drift Score={details['mean_drift_score']:.2f}, Is Drifted={details['is_drifted']}")


    # Test performance degradation
    ref_metrics = {"accuracy": 0.90, "roc_auc": 0.95}
    # Dummy predict function for testing
    def dummy_predict(df): return np.random.randint(0,2,size=len(df))

    logger.info("\nChecking performance degradation:")
    perf_check = monitor.check_model_performance_degradation(
        "PDModel",
        pd.DataFrame({'feature1': [1,2,3], 'target': [0,1,0]}), # Dummy validation data
        dummy_predict,
        ref_metrics,
        metric_to_check="accuracy"
    )
    logger.info(f"Performance degradation check results: {perf_check}")

    # Clean up test log file
    test_log = Path(monitor.monitoring_log_path)
    if test_log.exists():
        test_log.unlink()
        logger.info(f"\nCleaned up test monitoring log: {test_log}")
