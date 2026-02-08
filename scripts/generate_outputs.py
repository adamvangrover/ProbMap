import json
import datetime
from pathlib import Path
import logging

# Assuming src is in PYTHONPATH or this script is run from project root
from src.mlops.model_registry import ModelRegistry
from src.core.config import settings # To get MODEL_ARTIFACT_PATH for registry

# Imports for generate_json_ld_output
from src.data_management.knowledge_base import KnowledgeBaseService
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_graph import KnowledgeGraphService
from src.risk_map.risk_map_service import RiskMapService

# Simulation imports
import torch
import numpy as np
from src.simulation.oswm import OneShotWorldModel, SyntheticCreditPrior, OSWMTrainer

logger = logging.getLogger(__name__)
# Configure logging only if no handlers are already set up (e.g., by other imports or a global config)
if not logger.hasHandlers() and not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define project root if running script from within scripts/ directory
# This helps ensure paths are relative to the project root.
PROJECT_ROOT = Path(__file__).parent.parent

def generate_orchestration_manifest():
    """
    Generates an orchestration manifest JSON file that provides a snapshot
    of the system's key components, data sources, and active models.
    """
    logger.info("Starting generation of orchestration_manifest.json...")

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "orchestration_manifest.json"

    system_version = "1.0.0-final-demo"
    generation_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    data_sources = [
        {"name": "Company Data", "path": "data/sample_companies.csv", "description": "CSV file containing sample corporate entity data."},
        {"name": "Loan Agreements", "path": "data/sample_loans.json", "description": "JSON file containing sample loan agreement data."},
        {"name": "Financial Statements", "path": "data/sample_financial_statements.json", "description": "JSON file containing sample financial statements."},
        {"name": "Default Events", "path": "data/sample_default_events.json", "description": "JSON file containing sample default event data."},
        {"name": "PD Model Reference Features", "path": "data/reference_pd_features.csv", "description": "CSV file with reference features for PD model data drift detection."}
    ]

    active_models_list = []
    try:
        registry = ModelRegistry() # Uses default registry path from settings

        pd_model_details = registry.get_latest_model("PDModel", status="production")
        if pd_model_details:
            active_models_list.append({
                "model_name": "PDModel",
                "version": pd_model_details.get("model_version", "N/A"),
                "path": pd_model_details.get("model_path", "N/A"),
                "status": "production"
            })
        else:
            logger.warning("No production PDModel found in registry. It will be omitted from active_models.")

        lgd_model_details = registry.get_latest_model("LGDModel", status="production")
        if lgd_model_details:
            active_models_list.append({
                "model_name": "LGDModel",
                "version": lgd_model_details.get("model_version", "N/A"),
                "path": lgd_model_details.get("model_path", "N/A"),
                "status": "production"
            })
        else:
            logger.warning("No production LGDModel found in registry. It will be omitted from active_models.")

    except Exception as e:
        logger.error(f"Could not access model registry or fetch models: {e}. Active models list may be incomplete.")


    key_services = [
        {"service_name": "KnowledgeBaseService", "module_path": "src.data_management.knowledge_base"},
        {"service_name": "KnowledgeGraphService", "module_path": "src.data_management.knowledge_graph"},
        {"service_name": "PDModel", "module_path": "src.risk_models.pd_model"},
        {"service_name": "LGDModel", "module_path": "src.risk_models.lgd_model"},
        {"service_name": "PricingModel", "module_path": "src.risk_models.pricing_model"},
        {"service_name": "RiskMapService", "module_path": "src.risk_map.risk_map_service"},
        {"service_name": "ModelRegistry", "module_path": "src.mlops.model_registry"},
        {"service_name": "ModelMonitor", "module_path": "src.mlops.monitoring"},
        {"service_name": "ScenarioGenerator", "module_path": "src.simulation.scenario_generator"},
        {"service_name": "StressTester", "module_path": "src.simulation.stress_tester"}
    ]

    # Initially, only the manifest itself is an output file. This can be expanded.
    output_files_list = [
        {"name": "Orchestration Manifest", "path": str(manifest_path.relative_to(PROJECT_ROOT)).replace('\\', '/'), "description": "This manifest file, providing a snapshot of system components and outputs."},
        {"name": "JSON-LD Risk Profiles", "path": "output/risk_profiles.jsonld", "description": "Sample company risk profiles in JSON-LD format, generated from the risk map overview."},
        {
            "name": "System Overview HTML Page",
            "path": "index.html", # Relative to project root
            "description": "Main HTML overview page with links and embedded notebook snapshots."
        },
        {
            "name": "Conceptual Portfolio Risk/Return Plot",
            "path": "output/plot_portfolio_risk_return.png",
            "description": "Static plot from analysis notebook showing conceptual portfolio risk vs. return."
        },
        {
            "name": "Conceptual Peer Comparison Plot",
            "path": "output/plot_peer_comparison.png",
            "description": "Static plot from analysis notebook showing conceptual company vs. peer metrics."
        },
        {
            "name": "Consolidated Data Export (Excel)",
            "path": "output/consolidated_data_export.xlsx",
            "description": "Excel file with key data tables: Portfolio Overview, Sector Summary, Country Summary, Companies KB, Synthetic Equities & Commodities."
        },
        {
            "name": "Excel Data Exporter Notebook",
            "path": "notebooks/03_excel_data_exporter.ipynb",
            "description": "Jupyter notebook to generate and preview the consolidated Excel data export."
        },
        {
            "name": "Final Excel Summary (Markdown)",
            "path": "output/final_excel_summary.md",
            "description": "Markdown file summarizing the content and structure of the consolidated_data_export.xlsx."
        }

    ]

    manifest = {
        "system_version": system_version,
        "generation_timestamp": generation_timestamp,
        "data_sources": data_sources,
        "active_models": active_models_list,
        "key_services": key_services,
        "output_files": output_files_list # Will be updated if other outputs are generated by this script
    }

    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Orchestration manifest generated successfully at: {manifest_path}")
    except IOError as e:
        logger.error(f"Error writing orchestration manifest to file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during manifest generation: {e}")

def generate_simulation_data():
    """
    Generates simulation_data.json using One-Shot World Model projections.
    This file is intended for the 3D visualization.
    """
    logger.info("Starting generation of simulation_data.json...")
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    sim_data_path = output_dir / "simulation_data.json"

    try:
        kb_service = KnowledgeBaseService()
        loans = kb_service.get_all_loans()
        companies_list = kb_service.get_all_companies()
        companies = {c.company_id: c for c in companies_list}

        if not loans:
            logger.warning("No loans found in KB. Skipping simulation generation.")
            return

        # Initialize OSWM
        # In a real scenario, load pretrained weights.
        state_dim = 10
        oswm = OneShotWorldModel(state_dim=state_dim, d_model=32)

        # Simple pre-training to ensure non-random walks (optional but good for demo)
        prior = SyntheticCreditPrior(state_dim=state_dim, hidden_dim=32)
        trainer = OSWMTrainer(oswm, prior)
        logger.info("Pre-training OSWM for simulation generation...")
        for _ in range(50):
            trainer.train_step(batch_size=8, sequence_length=20)

        oswm.eval()
        simulated_data = []

        for loan in loans:
            company = companies.get(loan.company_id)
            if not company: continue

            # Construct initial state vector (simplified mapping)
            # We map some loan/company features to the state vector
            # [Revenue, InterestRate, MgmtScore, ... random ...]
            # Normalize somewhat
            revenue = (company.revenue_usd_millions or 0) / 1000.0
            rate = (loan.interest_rate_percentage or 5.0) / 10.0
            # mgmt score not directly in company model, assuming random or fixed

            initial_state = torch.tensor([revenue, rate] + [0.0]*8, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 10)

            # Predict future (e.g. 1 year -> 12 steps)
            steps = 12
            trajectory = []
            current_input = initial_state

            with torch.no_grad():
                for _ in range(steps):
                    out = oswm(current_input)
                    next_s = out[:, -1, :].unsqueeze(1)
                    trajectory.append(next_s.squeeze().tolist())
                    current_input = torch.cat([current_input, next_s], dim=1)

            # The final state represents the projected state at maturity (or 1 year out)
            final_state = trajectory[-1]

            # We map back to "projected" metrics
            projected_rate = max(0, min(20, final_state[1] * 10.0)) # Denormalize

            # Structure for frontend
            sim_item = {
                "loan_id": loan.loan_id,
                "company_id": loan.company_id,
                "company_name": company.company_name,
                "loan_amount": loan.loan_amount,
                "currency": loan.currency.value if hasattr(loan.currency, 'value') else str(loan.currency),
                "maturity_date": loan.maturity_date.isoformat() if loan.maturity_date else None,
                "interest_rate_percentage": loan.interest_rate_percentage,
                "collateral_type": loan.collateral_type.value if hasattr(loan.collateral_type, 'value') else str(loan.collateral_type),
                "industry": company.industry_sector.value if hasattr(company.industry_sector, 'value') else str(company.industry_sector),
                "mgmt_score": 5.0 + (final_state[2] * 2.0), # specific mapping
                "projected_rate": projected_rate,
                "trajectory": trajectory
            }
            simulated_data.append(sim_item)

        with open(sim_data_path, 'w') as f:
            json.dump(simulated_data, f, indent=4)
        logger.info(f"simulation_data.json generated at {sim_data_path}")

    except Exception as e:
        logger.error(f"Failed to generate simulation data: {e}", exc_info=True)


def generate_json_ld_output():
    """
    Generates a sample JSON-LD file containing risk profiles for a few companies.
    """
    logger.info("Starting generation of risk_profiles.jsonld...")

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True) # Ensure output dir exists
    json_ld_path = output_dir / "risk_profiles.jsonld"

    try:
        # Initialize services
        logger.info("Initializing services for JSON-LD generation...")
        kb_service = KnowledgeBaseService()

        pd_model = PDModel()
        if not pd_model.load_model(): # load_model attempts to load production if default path fails
            logger.warning("PD Model could not be loaded. JSON-LD generation might use default PD values or fail if models are essential.")

        lgd_model = LGDModel()
        if not lgd_model.load_model():
            logger.warning("LGD Model could not be loaded. JSON-LD generation might use default LGD values or fail if models are essential.")

        # KG service is needed by RiskMapService
        kg_service = KnowledgeGraphService(kb_service=kb_service)

        risk_map_service = RiskMapService(
            kb_service=kb_service,
            pd_model=pd_model,
            lgd_model=lgd_model,
            kg_service=kg_service
        )
        logger.info("Services initialized.")

        # Fetch portfolio overview data
        portfolio_overview = risk_map_service.generate_portfolio_risk_overview()

        if not portfolio_overview:
            logger.warning("Portfolio overview is empty. Cannot generate JSON-LD profiles.")
            # Create an empty JSON-LD structure or one with just context
            empty_json_ld = {
                "@context": {
                    "schema": "https://schema.org/",
                    "cr": "http://example.com/creditriskontology#"
                    # Add other context terms if desired for an empty graph
                },
                "@graph": []
            }
            with open(json_ld_path, 'w') as f:
                json.dump(empty_json_ld, f, indent=4)
            logger.info(f"Empty JSON-LD file generated at: {json_ld_path} due to no portfolio data.")
            return

        # Select a small number of companies (e.g., first 3-5)
        sample_profiles = portfolio_overview[:min(len(portfolio_overview), 5)]
        logger.info(f"Selected {len(sample_profiles)} profiles for JSON-LD output.")

        # Define JSON-LD context
        context_dict = {
            "schema": "https://schema.org/",
            "cr": "http://example.com/creditriskontology#",
            "CorporateEntity": "cr:CorporateEntity",
            "company_id": "schema:identifier",
            "company_name": "schema:name",
            "industry_sector": "schema:industry",
            "country_iso_code": "schema:addressCountry",
            "hasRiskProfile": "cr:hasRiskProfile",
            "RiskProfile": "cr:RiskProfile",
            "pd_estimate": "cr:probabilityOfDefault",
            "lgd_estimate": "cr:lossGivenDefault",
            "expected_loss_usd": "cr:expectedLoss",
            "management_quality_score": "cr:managementQualityScore",
            "kg_degree_centrality": "cr:degreeCentrality",
            "kg_num_suppliers": "cr:numberOfSuppliers",
            "kg_num_customers": "cr:numberOfCustomers"
        }

        json_ld_entities = []
        for item in sample_profiles:
            # item is a RiskItem Pydantic model
            pd_est = getattr(item, "effective_pd_estimate", None) or getattr(item, "model_pd_estimate", None)
            lgd_est = getattr(item, "effective_lgd_estimate", None) or getattr(item, "model_lgd_estimate", None)
            el_est = getattr(item, "effective_expected_loss_usd", None) or getattr(item, "model_expected_loss_usd", None)
            mqs = getattr(item, "effective_management_quality_score", None) or getattr(item, "original_management_quality_score", None)

            risk_profile_data = {
                "@type": "RiskProfile",
                "pd_estimate": pd_est,
                "lgd_estimate": lgd_est,
                "expected_loss_usd": el_est if el_est != "N/A" else None,
                "management_quality_score": mqs if mqs != "N/A" else None,
                "kg_degree_centrality": getattr(item, "kg_degree_centrality", None),
                "kg_num_suppliers": getattr(item, "kg_num_suppliers", None),
                "kg_num_customers": getattr(item, "kg_num_customers", None)
            }
            # Remove None values from risk_profile_data to keep JSON-LD clean
            risk_profile_data = {k: v for k, v in risk_profile_data.items() if v is not None}


            entity = {
                "@type": "CorporateEntity",
                "company_id": getattr(item, "company_id", None),
                "company_name": getattr(item, "company_name", None),
                "industry_sector": getattr(item, "industry_sector", None),
                "country_iso_code": getattr(item, "country_iso_code", None),
                "hasRiskProfile": risk_profile_data
            }
            json_ld_entities.append(entity)

        final_json_ld_object = {
            "@context": context_dict,
            "@graph": json_ld_entities
        }

        with open(json_ld_path, 'w') as f:
            json.dump(final_json_ld_object, f, indent=4)
        logger.info(f"JSON-LD risk profiles generated successfully at: {json_ld_path}")

    except Exception as e:
        logger.error(f"An error occurred during JSON-LD generation: {e}", exc_info=True)


if __name__ == "__main__":
    # Ensure PYTHONPATH includes the project root if running this script directly,
    # e.g., by running from the project root as `python scripts/generate_outputs.py`
    # or by setting PYTHONPATH=.
    generate_orchestration_manifest()
    generate_json_ld_output()
    generate_simulation_data()
