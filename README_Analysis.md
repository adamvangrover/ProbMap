# Proprietary Credit Risk Probability Map - Evolved System

## 1. Introduction

This document describes an evolved Proof of Concept (PoC) for a dynamic corporate credit risk analysis system. The core goal of this system is to provide a more holistic and dynamic view of credit risk by integrating quantitative models, qualitative data, knowledge graph context, and scenario analysis.

The system operates on a synthetic, file-based dataset for demonstration purposes. It aims to showcase advanced analytical capabilities and a more robust MLOps foundation than a typical initial PoC.

## 2. System Architecture

The system is composed of several key logical components:

*   **Data Layer:**
    *   `Ontology (src.data_management.ontology)`: Defines core financial concepts and data structures using Pydantic models (e.g., `CorporateEntity`, `LoanAgreement`, `FinancialStatement`).
    *   `Knowledge Base (src.data_management.knowledge_base)`: Manages access to data stored in local CSV and JSON files (e.g., `sample_companies.csv`, `sample_loans.json`). It includes data loading, validation against the ontology, and provides an access interface for other services.
    *   `Knowledge Graph (src.data_management.knowledge_graph)`: Constructs and manages an in-memory graph (using NetworkX) representing entities (companies, loans, etc.) and their relationships. It's populated from the Knowledge Base and allows for network-based analysis (e.g., centrality, pathfinding).

*   **Modeling Layer:**
    *   `Risk Models (src.risk_models)`:
        *   **PDModel (Probability of Default):** Uses a `RandomForestClassifier` with enhanced feature engineering (financial ratios, time-based features, interaction terms). Includes SHAP explainability.
        *   **LGDModel (Loss Given Default):** Uses a `GradientBoostingRegressor` with features like collateral type, loan amount, seniority of debt, and economic condition indicators. Generates synthetic recovery rates for training.
        *   **PricingModel:** A rule-based model that calculates suggested loan interest rates based on PD, LGD, customer segment, and contextual factors from the knowledge graph.

*   **Analysis & Services Layer:**
    *   `Risk Map Service (src.risk_map)`: Aggregates individual loan/company risk profiles (PD, LGD, Expected Loss) and incorporates qualitative data (e.g., management quality score) and KG-derived metrics to create a portfolio-level risk overview. Provides summaries by sector and country.
    *   `Simulation (src.simulation)`:
        *   `ScenarioGenerator`: Creates scenarios by applying feature-level shocks (multiplicative, additive, override) to a baseline portfolio.
        *   `StressTester`: Applies generated scenarios to the portfolio, re-calculating PD and LGD using the risk models based on shocked features to assess impact on Expected Loss.

*   **MLOps Layer (`src.mlops`):**
    *   `Model Registry`: A JSON file-based registry (`model_registry.json`) to track trained model versions, paths, metrics, parameters, and status (e.g., "registered", "production"). Models (PD, LGD) are automatically registered after training.
    *   `Model Monitoring`: Includes conceptual prediction logging, data drift detection (for numerical and categorical features against a reference dataset), and simulated model performance degradation checks.

*   **API Layer (`src.api`):**
    *   A FastAPI application that provides conceptual endpoints for interacting with the system (e.g., fetching company info, calculating risk metrics, pricing loans). For this PoC, full interaction and demonstration are primarily through code and Jupyter notebooks.

*   **Outputs & Notebooks:**
    *   `Output Generation (scripts/generate_outputs.py)`: A script to produce an `orchestration_manifest.json` (cataloging data sources, models, services) and an example `risk_profiles.jsonld` file. The manifest is also updated by this script to include other key outputs like the Excel export.
    *   `Analysis Notebooks (notebooks/)`: Jupyter notebooks to demonstrate system usage, data exploration, model training, and comprehensive risk analysis.
        *   `01_data_and_pd_model_demo.ipynb`: Original PoC demo.
        *   `02_comprehensive_risk_analysis.ipynb`: The primary notebook showcasing the evolved system's capabilities.
        *   `03_excel_data_exporter.ipynb`: A notebook to generate a consolidated Excel export of key data tables.


## 3. Setup and Running the System

### 3.1. Prerequisites
*   Python 3.9+
*   It is highly recommended to use a Python virtual environment.

### 3.2. Installation
1.  Clone the repository:
    ```bash
    git clone <repository_url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <repository_name>
    ```
3.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```
4.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3.3. Training Models
The primary models (PD and LGD) can be trained by running their respective scripts directly. This is necessary if model artifacts (`.joblib` files) are not present or if you wish to retrain them. These scripts will use the sample data, save the trained model artifacts to the path defined in `src.core.config.settings.MODEL_ARTIFACT_PATH` (default: `./models_store`), and register them in the MLOps model registry (`./models_store/model_registry.json`).

```bash
python -m src.risk_models.pd_model
python -m src.risk_models.lgd_model
```
Subsequent runs of these scripts will overwrite previously trained models and register new versions. The `if __name__ == "__main__":` blocks in these scripts also promote the latest trained model to "production" status in the registry for demonstration purposes.

### 3.4. Running the API (Optional Demonstration)
The system includes a FastAPI application for conceptual API endpoints.
```bash
uvicorn src.api.main:app --reload --port 8000
```
The API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.

### 3.5. Generating System Outputs
To generate the orchestration manifest and an example JSON-LD file:
1.  Ensure models have been trained and registered (see section 3.3), as the script queries the model registry for "production" models.
2.  Run the script from the project root:
    ```bash
    python scripts/generate_outputs.py
    ```
3.  Outputs will be placed in the `output/` directory (`output/orchestration_manifest.json`, `output/risk_profiles.jsonld`).

### 3.6. Running Jupyter Notebooks
The `notebooks/` directory contains demonstrations and analysis:
*   `notebooks/01_data_and_pd_model_demo.ipynb`: The original PoC demo notebook. It might be slightly outdated compared to the latest enhancements but shows basic data loading and PD model training from that phase.
*   `notebooks/02_comprehensive_risk_analysis.ipynb`: The primary analysis notebook demonstrating the full capabilities of the evolved system.
*   `notebooks/03_excel_data_exporter.ipynb`: This notebook collects key data tables (portfolio overview, summaries, company data, synthetic asset examples) and exports them to `output/consolidated_data_export.xlsx`.


To run the notebooks:
1.  Ensure you have Jupyter Notebook or JupyterLab installed:
    ```bash
    pip install notebook jupyterlab
    ```
2.  Start Jupyter from the project root:
    ```bash
    jupyter notebook
    # OR
    jupyter lab
    ```
    Navigate to the `notebooks/` directory in the Jupyter interface.

## 4. Key Features & Capabilities

*   **Expanded Ontology & Rich Data Model:** Detailed Pydantic models for `CorporateEntity`, `LoanAgreement`, `FinancialStatement`, `DefaultEvent`, `Counterparty`, etc., providing a structured representation of complex financial data.
*   **Integrated Knowledge Graph:** Captures interconnections between entities (e.g., subsidiaries, suppliers, guarantors for loans). Provides contextual insights such as network centrality (`get_entity_centrality`) and relationship-based queries (`find_common_guarantors`).
*   **Advanced Risk Models:**
    *   **PD Model:** `RandomForestClassifier` leveraging enhanced features including financial ratios (Debt-to-Equity, Current Ratio, Net Profit Margin, ROE), time-based features (loan duration, company age at origination), and interaction terms.
    *   **LGD Model:** `GradientBoostingRegressor` incorporating loan features like collateral type, loan amount, seniority of debt, and an economic condition indicator.
*   **SHAP Explainability for PD Model:** Provides insights into the drivers of individual PD predictions using SHAP (SHapley Additive exPlanations).
*   **Dynamic Pricing Model:** Calculates suggested interest rates based on PD, LGD, and allows for adjustments based on customer segments and contextual information derived from the Knowledge Graph (e.g., centrality, supplier/customer network size).
*   **Comprehensive Risk Mapping:** The `RiskMapService` generates a multi-dimensional portfolio overview. Each risk item includes:
    *   Calculated PD, LGD, and Expected Loss (EL).
    *   Qualitative data like `management_quality_score`.
    *   KG-derived metrics such as degree centrality and counts of key relationships (suppliers, customers, subsidiaries).
    *   Aggregation of risk metrics by industry sector and country.
*   **MLOps Framework (PoC):**
    *   **Model Registry:** Automated registration of trained PD and LGD models (version, path, metrics, parameters, tags) into a JSON-based registry.
    *   **Production Model Loading:** Models can load the latest "production" version from the registry as a fallback if a specific path is not found.
    *   **Monitoring:**
        *   Conceptual prediction logging.
        *   Data drift detection for both numerical (mean, std dev) and categorical features (new categories, proportion changes) against a reference dataset.
        *   Simulated model performance degradation checks using logged predictions and synthetically generated ground truth.
*   **Sophisticated Scenario Simulation:**
    *   `ScenarioGenerator` applies feature-level shocks (multiplicative, additive, override) to raw input features of a portfolio.
    *   `StressTester` takes the shocked portfolio and re-calculates PD and LGD using the trained models to assess the impact on Expected Loss, providing a more realistic stress impact than directly shocking PD/LGD values.
*   **Consolidated Data Export:** Generates an Excel file (`output/consolidated_data_export.xlsx`) containing key data tables like portfolio overview, sector/country summaries, and company listings, suitable for external analysis in BI tools or spreadsheets. This is produced by the `notebooks/03_excel_data_exporter.ipynb` notebook.


## 5. Design Rationale & Logic

*   **Model Choices:**
    *   `RandomForestClassifier` (PD): Chosen for its ability to handle non-linear relationships, feature interactions, good general performance, and relative robustness to outliers. It also supports SHAP explainability.
    *   `GradientBoostingRegressor` (LGD): Selected for its strong predictive power, ability to capture complex patterns, and robustness. The target (recovery rate) is clipped to [0.05, 0.95] to keep it within a sensible range for regression.
*   **Feature Engineering:** The emphasis on creating financial ratios, time-based features (e.g., `company_age_at_origination`, `loan_duration_days`), and interaction terms aims to provide more predictive signals to the models than raw inputs alone.
*   **Knowledge Graph Utility:** The KG allows the system to move beyond analyzing entities in isolation. By representing and querying relationships (e.g., guarantors, subsidiaries, common suppliers), it uncovers network risks and contextual factors that can influence creditworthiness (e.g., concentration risks, systemic importance via centrality).
*   **Simulation Approach:** Applying shocks at the raw feature level (e.g., reducing company revenue, increasing market interest rates, worsening economic indicators) and then re-running the PD/LGD models provides a more realistic assessment of scenario impacts. This captures how changes in underlying conditions propagate through the models, rather than making arbitrary adjustments to final PD/LGD scores.
*   **MLOps Importance:** Even in a PoC, establishing a basic model registry and monitoring concepts (drift, performance) lays the groundwork for more robust and scalable MLOps practices. It emphasizes versioning, reproducibility, and awareness of model health over time.

## 6. Interpreting the "Probability Map" (Analysis Notebook)

The primary output and demonstration of the system's analytical capabilities are through the `notebooks/02_comprehensive_risk_analysis.ipynb` Jupyter notebook. This notebook effectively serves as the "Probability Map" by providing a multi-faceted view of risk.

*   **Portfolio Overview:** The notebook starts by loading all services and generating a portfolio-wide risk summary. This table gives an initial snapshot of PD, LGD, EL, and key contextual metrics for all entities.
*   **Aggregate Summaries:** Risk concentrations are shown by aggregating EL, PD, and LGD by sector and country, allowing for identification of high-risk segments.
*   **Deep Dives:** The core of the analysis lies in "deep dives" on selected synthetic companies. For each selected company, the notebook demonstrates:
    1.  **Data Compilation:** Fetches and displays all known data about the company: profile from KB, associated loans, latest financial statements, and KG-derived contextual info (centrality, number of connections).
    2.  **PD Model Explainability:** Uses SHAP to show the key features driving the PD model's prediction for that specific company/loan.
    3.  **Stress Test Impact:** Applies a targeted scenario (e.g., revenue shock) to the company's features and shows how its PD, LGD, and EL change, demonstrating its resilience.
    4.  **Narrative Summary:** A structured markdown section synthesizes all these data points (quantitative scores, financial health indicators, SHAP drivers, KG context, qualitative scores, stress test impact) into a holistic risk assessment. It includes placeholders for illustrative decision points and "probability band" considerations.
*   **"Public-Like" Synthetic Profiles:** The notebook further illustrates analysis on more complex, synthetically generated profiles with richer data (e.g., multiple financial statement periods) to show how the system would handle such cases.

Interpreting the "Probability Map" involves looking beyond a single PD score. For a given entity, an analyst would consider:
*   The **quantitative PD/LGD/EL scores**.
*   **Why** the PD model arrived at its score (via SHAP).
*   The company's **financial health** from its statements.
*   Its **network context** from the KG (is it overly reliant on few suppliers? Is it systemically important?).
*   **Qualitative factors** like management quality.
*   Its **resilience** to adverse scenarios via stress testing.

This multi-faceted approach allows for a more nuanced understanding of risk, especially for "grey zone" companies where different indicators might conflict.

## 7. Limitations

*   **Proof of Concept:** This system is a PoC and not a production-ready application.
*   **Synthetic Data:** All analysis and model training are based on a generated, synthetic dataset. There is no connection to real-world financial data, market feeds, or external APIs, which would be crucial for a production system.
*   **Simplified MLOps:** The model registry (JSON file) and monitoring components (conceptual checks, simulated ground truth) are highly simplified for demonstration and would require dedicated MLOps tools in a real deployment.
*   **No UI:** The "Probability Map" and detailed analysis are presented through Jupyter notebook outputs and console logs. A dedicated user interface for visualization and interactive exploration is not part of this PoC.
*   **Scalability:** While the design incorporates improvements, performance on very large datasets (millions of entities/relationships) has not been optimized. Database solutions for KB and KG, and distributed computing for modeling/simulations, would be needed for large-scale use.
*   **Model Scope:** The models are illustrative. Real-world PD/LGD models often require more extensive feature engineering, validation, and governance. The pricing model is also very basic.
*   **Static KG Context in Pricing:** The `PricingModel` currently takes KG context as a dictionary. In a more dynamic system, it might query the `KnowledgeGraphService` directly or receive more structured KG inputs.

## 8. Potential Future Directions

*   **Real Data Integration:** Connect to actual databases for company/loan data and external APIs for market data, financial news, etc.
*   **UI Development:** Create a web-based user interface for interactive risk exploration, "probability map" visualization, scenario definition, and reporting.
*   **Advanced MLOps:** Integrate with robust MLOps tools like MLflow, Kubeflow, Vertex AI, or SageMaker for experiment tracking, model versioning, deployment, and automated monitoring pipelines.
*   **Sophisticated Modeling:**
    *   Explore more complex machine learning models (e.g., Graph Neural Networks for leveraging the Knowledge Graph directly in PD/LGD, NLP for analyzing text data from news or company reports).
    *   Implement proper Beta Regression for LGD or other specialized financial risk models.
    *   Develop more granular and data-driven pricing models.
*   **Automated Reporting:** Generate standardized risk reports for entities, segments, or the entire portfolio.
*   **Enhanced Simulation:** More complex scenario definitions (e.g., macroeconomic model integration), correlated shocks, and more detailed impact analysis.
*   **Regulatory Compliance & Governance:** Incorporate features for model validation, audit trails, and compliance reporting.

## 9. Key Libraries / Sources

The system primarily utilizes the following Python libraries:
*   `fastapi`: For the API layer.
*   `uvicorn`: For running the FastAPI application.
*   `pydantic` & `pydantic-settings`: For data validation and settings management.
*   `python-dotenv`: For managing environment variables.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For machine learning models (RandomForest, GradientBoosting), preprocessing, and metrics.
*   `networkx`: For creating and analyzing the knowledge graph.
*   `joblib`: For saving and loading trained model artifacts.
*   `shap`: For PD model explainability.

Standard machine learning and data processing practices have been followed. Specific financial modeling methodologies are highly simplified for this PoC.
