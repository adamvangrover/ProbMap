# One-Shot World Model (OSWM) and Model Monitoring

This document details the recent additive enhancements to the Credit Risk System, specifically the implementation of the One-Shot World Model (OSWM) for simulation and the Model Monitoring service.

## 1. One-Shot World Model (OSWM)

### Overview
The OSWM is a Transformer-based model designed to simulate future credit states based on a history of context. It leverages "in-context learning," meaning it can adapt its predictions to the specific dynamics of a new environment (e.g., a specific company's credit trajectory) simply by being conditioned on a short history of that environment, without needing weight updates.

### Architecture
*   **Core:** `src/simulation/oswm.py`
    *   **SyntheticCreditPrior:** Generates synthetic episodes where the underlying dynamical laws (simulated by random MLPs) vary per episode. This forces the model to learn *how to learn* dynamics from context.
    *   **OneShotWorldModel:** A Causal Transformer that predicts the next state vector given a sequence of previous states.
*   **Usage:**
    *   **API:** `POST /api/v1/simulate-trajectory`
        *   Accepts `initial_state` and optional `context_data`.
        *   Returns a simulated `trajectory` of future states.
    *   **Simulation:** `scripts/generate_outputs.py` uses OSWM to generate `simulation_data.json`, which drives the 3D visualization.

### Key Features
*   **Adaptation:** Adapts to new "regimes" (volatility, trend, mean reversion) purely from input context.
*   **Efficiency:** Inference-time adaptation is fast (single forward pass).

## 2. Model Monitoring

### Overview
The monitoring service tracks model predictions in production to detect data drift, concept drift, and performance degradation.

### Components
*   **Monitor:** `src/mlops/monitoring.py` -> `ModelMonitor` class.
*   **Logging:** Predictions from the API (`PDModel` and `LGDModel`) are logged to a JSONL file (conceptually, or to a real store).
*   **Drift Detection:**
    *   **Data Drift:** Compares statistical properties (mean, std, categorical distribution) of current production data against a reference dataset.
    *   **Concept Drift (Proxy):** Monitors shifts in the distribution of predicted probabilities/outputs.
*   **Performance:** Can simulate ground truth (or ingest it) to check metrics like Accuracy and ROC AUC over time.

### Integration
*   The `calculate_risk_metrics` endpoint automatically logs input features and model outputs to the monitoring system.

## 3. Future Directions
*   **OSWM:** Train on real historical credit data instead of a synthetic prior to capture realistic macro-economic regimes.
*   **Monitoring:** Integrate with tools like Evidently AI or Prometheus for real-time dashboards and alerting.
