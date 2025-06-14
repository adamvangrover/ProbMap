# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Notebook: Data Loading and PD Model Demonstration
#
# This notebook demonstrates:
# 1. Initializing the KnowledgeBaseService to load sample data.
# 2. Accessing company and loan data.
# 3. Initializing and training a Probability of Default (PD) model.
# 4. Making a prediction with the trained PD model.
# 5. (Conceptual) Interacting with the KnowledgeGraphService.

# %% [markdown]
# ## 1. Imports and Setup
#
# Ensure your Python environment is set up with all dependencies from `requirements.txt`.
# It's best to run this notebook from the root of the project directory so that imports work correctly.
# You might need to adjust `PYTHONPATH` if running from elsewhere or add `src` to `sys.path`.

# %%
import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add src to Python path if not already there (common for notebooks)
project_root = Path(os.getcwd()).parent # Assumes notebook is in 'notebooks' folder, so parent is root
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging (optional, but good for seeing what services are doing)
# If core.logging_config is imported by other modules, it might already be set up.
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NotebookDemo")

# %% [markdown]
# ## 2. Initialize Services

# %%
from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.knowledge_graph import KnowledgeGraphService
from src.risk_models.pd_model import PDModel
from src.core.config import settings # To see where models might be stored

logger.info(f"Model artifacts are expected to be in: {settings.MODEL_ARTIFACT_PATH}")

# %%
logger.info("Initializing KnowledgeBaseService...")
kb_service = KnowledgeBaseService() # Loads sample_companies.csv and sample_loans.json by default

# %% [markdown]
# ## 3. Accessing Data from Knowledge Base

# %%
logger.info("Fetching all companies from Knowledge Base...")
all_companies = kb_service.get_all_companies()
if all_companies:
    logger.info(f"Found {len(all_companies)} companies.")
    # Display first few companies as a DataFrame for better readability in notebook
    companies_df = pd.DataFrame([c.model_dump() for c in all_companies])
    print("Sample Companies:")
    print(companies_df.head())
else:
    logger.warning("No companies found in Knowledge Base.")

# %%
logger.info("Fetching all loans from Knowledge Base...")
all_loans = kb_service.get_all_loans()
if all_loans:
    logger.info(f"Found {len(all_loans)} loans.")
    loans_df = pd.DataFrame([l.model_dump() for l in all_loans])
    print("\nSample Loans:")
    print(loans_df.head())
else:
    logger.warning("No loans found in Knowledge Base.")

# %%
# Get a specific company and its loans
target_company_id = "COMP001" # From sample_companies.csv
logger.info(f"Fetching profile for company: {target_company_id}")
company_profile = kb_service.get_company_profile(target_company_id)
if company_profile:
    print(f"\nProfile for {target_company_id}:")
    print(pd.Series(company_profile.model_dump()))

    logger.info(f"Fetching loans for company: {target_company_id}")
    company_loans = kb_service.get_loans_for_company(target_company_id)
    if company_loans:
        print(f"\nLoans for {target_company_id}:")
        print(pd.DataFrame([l.model_dump() for l in company_loans]))
    else:
        logger.info(f"No loans found for {target_company_id}.")
else:
    logger.warning(f"Company {target_company_id} not found.")


# %% [markdown]
# ## 4. PD Model Training and Prediction
#
# We'll initialize the PDModel. If a pre-trained model exists at the configured path, it might be loaded. Otherwise, we'll train it using the data from our Knowledge Base.

# %%
logger.info("Initializing PDModel...")
pd_model = PDModel() # Uses settings.MODEL_ARTIFACT_PATH for model file

# Attempt to load a pre-trained model
if pd_model.load_model():
    logger.info(f"PD Model loaded successfully from {pd_model.model_path}")
else:
    logger.warning(f"No pre-trained PD model found at {pd_model.model_path} or failed to load. Training a new one...")
    if not all_companies or not all_loans:
        logger.error("Cannot train PD model: Company or loan data is missing from Knowledge Base.")
    else:
        training_metrics = pd_model.train(kb_service=kb_service)
        logger.info(f"PD Model training complete. Metrics: {training_metrics}")
        if "error" in training_metrics:
             logger.error(f"PD Model training resulted in error: {training_metrics['error']}")


# %% [markdown]
# ### Make a Prediction
# We need data for a loan and its associated company to make a prediction.

# %%
if pd_model.model is None:
    logger.error("PD Model is not available (not loaded or trained). Cannot make predictions.")
else:
    if all_loans and all_companies:
        # Use the first loan and its company for prediction
        sample_loan_for_pred = all_loans[0]
        sample_company_for_pred = kb_service.get_company_profile(sample_loan_for_pred.company_id)

        if sample_company_for_pred:
            logger.info(f"Predicting PD for loan: {sample_loan_for_pred.loan_id} (Company: {sample_company_for_pred.company_id})")

            # The predict_for_loan method expects dictionaries
            loan_dict = sample_loan_for_pred.model_dump()
            company_dict = sample_company_for_pred.model_dump()

            prediction_result = pd_model.predict_for_loan(loan_dict, company_dict)

            if prediction_result:
                pred_class, pred_probability = prediction_result
                print(f"\nPD Prediction for Loan {sample_loan_for_pred.loan_id}:")
                print(f"  Predicted Class (1=Default, 0=Non-Default): {pred_class}")
                print(f"  Probability of Default: {pred_probability:.4f}")
            else:
                logger.error("PD prediction failed for the sample loan.")
        else:
            logger.warning(f"Could not find company details for loan {sample_loan_for_pred.loan_id} to make a prediction.")
    else:
        logger.warning("Not enough data (loans/companies) to select a sample for PD prediction.")

# %% [markdown]
# ## 5. Knowledge Graph Interaction (Conceptual)
# Initialize the KnowledgeGraphService. If a KnowledgeBaseService is provided, it will attempt to populate the graph.

# %%
logger.info("Initializing KnowledgeGraphService...")
# This will populate the graph using the data already loaded by kb_service
kg_service = KnowledgeGraphService(kb_service=kb_service)

logger.info(f"Knowledge Graph created with {kg_service.graph.number_of_nodes()} nodes and {kg_service.graph.number_of_edges()} edges.")

# %%
# Example: Get neighbors of a company in the graph
if company_profile: # Using company_profile from earlier cell
    company_node_id = company_profile.company_id
    logger.info(f"Getting neighbors of company {company_node_id} in the Knowledge Graph:")

    # Neighbors with any relationship type
    all_neighbors = kg_service.get_neighbors(company_node_id)
    print(f"  All neighbors of {company_node_id}: {all_neighbors}")

    # Neighbors with a specific relationship type (e.g., HAS_LOAN)
    # Note: RelationshipType is defined in src.data_management.knowledge_graph
    from src.data_management.knowledge_graph import RelationshipType
    loan_neighbors = kg_service.get_neighbors(company_node_id, relationship_type=RelationshipType.HAS_LOAN)
    print(f"  Loans associated with {company_node_id}: {loan_neighbors}")

    sector_neighbors = kg_service.get_neighbors(company_node_id, relationship_type=RelationshipType.LOCATED_IN_SECTOR)
    if sector_neighbors:
        print(f"  Sector for {company_node_id}: {sector_neighbors[0]}")


# %%
# Example: Get default history for a company
defaulting_company_id = "COMP004" # This company has a defaulted loan in sample data
logger.info(f"Getting default history for company: {defaulting_company_id}")
default_history = kg_service.get_company_default_history(defaulting_company_id)
if default_history:
    print(f"Default history for {defaulting_company_id}:")
    for event in default_history:
        print(f"  - Loan ID: {event['loan_id']}, Default Date: {event.get('default_date', 'N/A')}, Amount: {event.get('loan_amount')}")
else:
    print(f"No default history found for {defaulting_company_id} in the graph (or company not found).")


# %% [markdown]
# ## End of Demonstration
# This notebook covered basic interactions with the data management layer and one of the risk models. Further notebooks could explore:
# - LGD model training and prediction.
# - Pricing model usage.
# - Risk Map Service data generation.
# - Scenario simulation and stress testing outputs.
