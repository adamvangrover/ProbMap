# AGENTS.md - Risk Model Development (`src/risk_models/`)

This document provides specific guidelines for developing, enhancing, and maintaining risk models (PD, LGD, Pricing, and any future models) within the `src/risk_models/` directory. It is an extension of the guidelines in `src/AGENTS.md` and the root `AGENTS.md`.

## 1. Model Development Lifecycle

*   **Prototyping:** Experiment with new model types, features, and libraries in notebooks (`notebooks/`) before integrating them into `src/risk_models/`.
*   **Training (`train` method):**
    *   Each model class MUST have a `train` method.
    *   This method should encapsulate all steps required to train the model: data preparation, feature engineering, model fitting, and evaluation.
    *   It MUST return a dictionary of key performance metrics.
    *   It MUST save the trained model artifact (e.g., using `joblib` or native model saving methods) to the path specified in `core.config.settings.MODEL_ARTIFACT_PATH`.
    *   It MUST register the trained model (version, path, metrics, parameters, tags) with the `ModelRegistry` (`src.mlops.model_registry.py`).
*   **Prediction (`predict` or similar methods):**
    *   Each model class MUST have one or more methods for making predictions (e.g., `predict`, `predict_proba`, `predict_lgd`).
    *   These methods should clearly define their expected input format (e.g., Pandas DataFrame, dictionary of features) and output format.
    *   Input data for prediction should undergo the same preprocessing and feature engineering steps as the training data.
*   **Model Loading (`load_model` method):**
    *   Each model class MUST have a `load_model` method.
    *   This method should be able to load a pre-trained model artifact from a specified path.
    *   It SHOULD implement fallback logic to load the latest "production" version of the model from the `ModelRegistry` if a specific path is not provided or if loading from the path fails.
*   **Saving (`save_model` method):**
    *   Each model class MUST have a `save_model` method to persist the trained model object.

## 2. Feature Engineering

*   **Encapsulation:** Feature engineering logic specific to a model should ideally be encapsulated within the model's class (e.g., in a private `_prepare_features` method) or in a dedicated feature engineering module that the model class uses.
*   **Reproducibility:** Ensure that feature engineering steps are applied consistently during both training and prediction.
*   **Documentation:**
    *   Clearly document all engineered features: their definition, rationale, and how they are computed.
    *   Maintain a list of raw input features expected by the model.
*   **HITL-Derived Features:** If features are derived from HITL inputs (e.g., validated qualitative scores), ensure these are clearly marked and their provenance is traceable.
*   **New Data Dimensions:** When incorporating new data dimensions (e.g., from external data sources), document how these are transformed into features.

## 3. Model Interfaces and Structure

*   **Base Class (Optional):** Consider creating a base `RiskModel` class if common functionalities (e.g., loading/saving, registration logic) can be abstracted.
*   **Configuration:** Model hyperparameters and other settings (e.g., feature lists) should be configurable, potentially via `src.core.config` or passed during model instantiation.
*   **Explainability:**
    *   Models (especially complex ones like tree ensembles or neural networks) SHOULD provide mechanisms for explainability (e.g., SHAP value calculation, feature importance plots).
    *   The `PDModel`'s `get_feature_importance_shap` method is an example. Strive for similar capabilities in other models.
*   **Uncertainty Quantification:** Where feasible and appropriate for the model type, provide methods to estimate the uncertainty of predictions (e.g., prediction intervals, probability distributions).

## 4. Human-in-the-Loop (HITL) Integration for Models

*   **Consuming Feedback for Retraining:**
    *   Design models and training pipelines to potentially consume curated HITL feedback (e.g., corrected labels, instances flagged as misclassified with high confidence by analysts). This feedback can be used to create weighted samples, augment training datasets, or for targeted fine-tuning.
    *   The format and mechanism for accessing this feedback will be coordinated with `KnowledgeBaseService` and MLOps components.
*   **Output for HITL Review:**
    *   Model prediction methods should output information in a way that facilitates HITL review. This includes not just the prediction, but also probabilities, confidence scores, or uncertainty estimates.
    *   SHAP values or other explainability outputs are crucial for this.
*   **Model Evaluation with HITL Insights:**
    *   Beyond standard metrics, evaluate models based on their agreement with expert judgment on critical cases or segments identified by HITL.

## 5. Performance Metrics and Evaluation

*   **Standard Metrics:** Report standard performance metrics relevant to the model type (e.g., Accuracy, Precision, Recall, F1-score, ROC AUC for classification; MSE, MAE, R-squared for regression).
*   **Business-Relevant Metrics:** Where possible, also track metrics that are more directly interpretable in a business context (e.g., impact of model errors on overall portfolio EL).
*   **Bias and Fairness:** (Future) As the system matures, incorporate checks for model bias and fairness across different demographic groups or segments if relevant data becomes available.
*   **Validation Sets:** Ensure proper use of training, validation, and test sets to avoid overfitting and get a realistic estimate of out-of-sample performance.

## 6. MLOps Integration (`src.mlops/`)

*   **Model Registry (`ModelRegistry`):**
    *   All trained models MUST be registered in the `ModelRegistry`.
    *   Ensure all required metadata (model name, version, path, metrics, parameters, tags) is accurately logged.
    *   Use meaningful tags (e.g., `model_type: RandomForestClassifier`, `stage: training`, `dataset_version: v1.2`).
*   **Model Monitoring (`monitoring.py`):**
    *   (Future) As monitoring capabilities are built out (e.g., data drift detection, prediction drift, performance degradation checks), models may need to output logs or data in a format consumable by these monitoring tools.
    *   HITL should be involved in reviewing and interpreting alerts from the monitoring system.

Adherence to these guidelines will help ensure that risk models are robust, well-documented, maintainable, and effectively integrated into the AI HITL framework.
