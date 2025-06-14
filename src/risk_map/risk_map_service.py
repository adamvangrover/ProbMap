import logging
from typing import List, Dict, Any, Optional

from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import LoanAgreement, CorporateEntity
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_graph import KnowledgeGraphService

logger = logging.getLogger(__name__)

class RiskMapService:
    """
    Service to generate data for the dynamic corporate risk rating map.
    For PoC, this focuses on aggregating risk metrics (PD, LGD, EL)
    for a portfolio of loans/companies. Visualization is out of scope for this PoC.
    """
    def __init__(self,
                 kb_service: KnowledgeBaseService,
                 pd_model: PDModel,
                 lgd_model: LGDModel,
                 kg_service: Optional[KnowledgeGraphService] = None
                 ):
        self.kb_service = kb_service
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.kg_service = kg_service

        # Ensure models are loaded if they are not already trained/in memory
        if self.pd_model.model is None:
            logger.info("PD model not loaded, attempting to load from default path.")
            if not self.pd_model.load_model():
                logger.warning("PD model could not be loaded. Risk map generation might be impaired or use defaults.")
                # As a fallback for PoC if model loading fails, we might need to train it here,
                # or ensure it's trained and saved in a previous step.
                # For now, we assume it should have been trained/loaded.
                # Consider: pd_model.train(kb_service) # Potentially train if not loaded.

        if self.lgd_model.model is None:
            logger.info("LGD model not loaded, attempting to load from default path.")
            if not self.lgd_model.load_model():
                logger.warning("LGD model could not be loaded. Risk map generation might be impaired or use defaults.")
                # Consider: lgd_model.train(kb_service) # Potentially train if not loaded.


    def generate_portfolio_risk_overview(self) -> List[Dict[str, Any]]:
        """
        Generates a risk overview for the entire portfolio available in the Knowledge Base.
        Each item in the list represents a loan with its associated risk metrics.
        """
        logger.info("Generating portfolio risk overview for the risk map...")
        portfolio_risk_data = []

        all_loans = self.kb_service.get_all_loans()
        if not all_loans:
            logger.warning("No loans found in Knowledge Base. Cannot generate risk map data.")
            return []

        for loan in all_loans:
            company = self.kb_service.get_company_profile(loan.company_id)
            if not company:
                logger.warning(f"Company {loan.company_id} for loan {loan.loan_id} not found. Skipping loan in risk map.")
                continue

            # 1. Get PD estimate
            pd_prediction_class, pd_probability = -1, -1.0 # Default/Error values
            if self.pd_model.model is not None:
                pd_result = self.pd_model.predict_for_loan(loan.model_dump(), company.model_dump())
                if pd_result:
                    pd_prediction_class, pd_probability = pd_result
                else:
                    logger.warning(f"Could not get PD prediction for loan {loan.loan_id}. Using default.")
                    pd_probability = 0.5 # Default PD if prediction fails
            else:
                logger.warning(f"PD model not available for loan {loan.loan_id}. Using default PD.")
                pd_probability = 0.5 # Default PD

            # 2. Get LGD estimate
            lgd_estimate = -1.0 # Default/Error value
            if self.lgd_model.model is not None:
                # LGD model expects a dictionary of features
                # Align with features expected by the enhanced LGDModel
                lgd_features = {
                    'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None',
                    'loan_amount_usd': loan.loan_amount,
                    'seniority_of_debt': str(loan.seniority_of_debt) if loan.seniority_of_debt else 'Unknown',
                    'economic_condition_indicator': loan.economic_condition_indicator if loan.economic_condition_indicator is not None else 0.5
                }
                lgd_estimate = self.lgd_model.predict_lgd(lgd_features)
            else:
                logger.warning(f"LGD model not available for loan {loan.loan_id}. Using default LGD.")
                lgd_estimate = 0.75 # Default LGD

            # 3. Calculate Expected Loss (EL)
            ead = loan.loan_amount
            expected_loss = pd_probability * lgd_estimate * ead if pd_probability >=0 and lgd_estimate >=0 else -1

            risk_item = {
                "loan_id": loan.loan_id,
                "company_id": company.company_id,
                "company_name": company.company_name,
                "industry_sector": company.industry_sector.value if company.industry_sector else 'N/A',
                "country_iso_code": company.country_iso_code if company.country_iso_code else 'N/A',
                "management_quality_score": company.management_quality_score if hasattr(company, 'management_quality_score') and company.management_quality_score is not None else 'N/A',
                "loan_amount_usd": loan.loan_amount,
                "pd_estimate": round(pd_probability, 4),
                "lgd_estimate": round(lgd_estimate, 4),
                "exposure_at_default_usd": ead,
                "expected_loss_usd": round(expected_loss, 2) if expected_loss >=0 else "N/A",
                "currency": loan.currency.value,
                "collateral_type": loan.collateral_type.value if loan.collateral_type else "None",
                "is_defaulted": loan.default_status,
                # KG-derived fields (default to N/A)
                "kg_degree_centrality": "N/A",
                "kg_num_suppliers": "N/A",
                "kg_num_customers": "N/A",
                "kg_num_subsidiaries": "N/A"
            }

            # Add more contextual info from Knowledge Graph if available
            if self.kg_service and self.kg_service.graph.has_node(company.company_id):
                context_info = self.kg_service.get_company_contextual_info(company.company_id)
                if context_info:
                    risk_item["kg_degree_centrality"] = round(context_info.get('degree_centrality', -1.0), 4) if context_info.get('degree_centrality', -1.0) != -1.0 else "N/A"
                    risk_item["kg_num_suppliers"] = context_info.get('num_suppliers', 'N/A')
                    risk_item["kg_num_customers"] = context_info.get('num_customers', 'N/A')
                    risk_item["kg_num_subsidiaries"] = context_info.get('num_subsidiaries', 'N/A')

            portfolio_risk_data.append(risk_item)

        logger.info(f"Generated risk overview for {len(portfolio_risk_data)} loans.")
        return portfolio_risk_data

    def get_risk_summary_by_sector(self, portfolio_overview: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Aggregates risk metrics by industry sector.
        """
        if portfolio_overview is None:
            portfolio_overview = self.generate_portfolio_risk_overview()

        sector_summary: Dict[str, Dict[str, Any]] = {}

        for item in portfolio_overview:
            sector = item["industry_sector"]
            if sector not in sector_summary:
                sector_summary[sector] = {
                    "total_exposure": 0.0,
                    "total_expected_loss": 0.0,
                    "loan_count": 0,
                    "average_pd": [], # Store individual PDs to average later
                    "average_lgd": [], # Store individual LGDs to average later
                    "defaulted_loan_count": 0
                }

            current_el = item["expected_loss_usd"] if item["expected_loss_usd"] != "N/A" else 0

            sector_summary[sector]["total_exposure"] += item["exposure_at_default_usd"]
            sector_summary[sector]["total_expected_loss"] += current_el
            sector_summary[sector]["loan_count"] += 1
            if item["pd_estimate"] >= 0: # Only average valid estimates
                 sector_summary[sector]["average_pd"].append(item["pd_estimate"])
            if item["lgd_estimate"] >= 0: # Only average valid estimates
                sector_summary[sector]["average_lgd"].append(item["lgd_estimate"])
            if item["is_defaulted"]:
                sector_summary[sector]["defaulted_loan_count"] +=1


        # Calculate averages
        for sector, data in sector_summary.items():
            avg_pd = sum(data["average_pd"]) / len(data["average_pd"]) if data["average_pd"] else 0
            avg_lgd = sum(data["average_lgd"]) / len(data["average_lgd"]) if data["average_lgd"] else 0
            sector_summary[sector]["average_pd"] = round(avg_pd, 4)
            sector_summary[sector]["average_lgd"] = round(avg_lgd, 4)
            sector_summary[sector]["total_expected_loss"] = round(data["total_expected_loss"], 2)


        logger.info(f"Generated risk summary for {len(sector_summary)} sectors.")
        return sector_summary

    def get_risk_summary_by_country(self, portfolio_overview: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Aggregates risk metrics by country.
        """
        if portfolio_overview is None:
            portfolio_overview = self.generate_portfolio_risk_overview()

        country_summary: Dict[str, Dict[str, Any]] = {}

        for item in portfolio_overview:
            country_code = item.get("country_iso_code", "Unknown") # Use .get() for safety
            if country_code not in country_summary:
                country_summary[country_code] = {
                    "total_exposure": 0.0,
                    "total_expected_loss": 0.0,
                    "loan_count": 0,
                    "average_pd": [],
                    "average_lgd": [],
                    "defaulted_loan_count": 0
                }

            current_el = item["expected_loss_usd"] if item["expected_loss_usd"] != "N/A" else 0

            country_summary[country_code]["total_exposure"] += item["exposure_at_default_usd"]
            country_summary[country_code]["total_expected_loss"] += current_el
            country_summary[country_code]["loan_count"] += 1
            if item["pd_estimate"] >= 0:
                 country_summary[country_code]["average_pd"].append(item["pd_estimate"])
            if item["lgd_estimate"] >= 0:
                country_summary[country_code]["average_lgd"].append(item["lgd_estimate"])
            if item["is_defaulted"]:
                country_summary[country_code]["defaulted_loan_count"] +=1

        # Calculate averages
        for country_code, data in country_summary.items():
            avg_pd = sum(data["average_pd"]) / len(data["average_pd"]) if data["average_pd"] else 0
            avg_lgd = sum(data["average_lgd"]) / len(data["average_lgd"]) if data["average_lgd"] else 0
            country_summary[country_code]["average_pd"] = round(avg_pd, 4)
            country_summary[country_code]["average_lgd"] = round(avg_lgd, 4)
            country_summary[country_code]["total_expected_loss"] = round(data["total_expected_loss"], 2)

        logger.info(f"Generated risk summary for {len(country_summary)} countries.")
        return country_summary


if __name__ == "__main__":
    # Setup for standalone testing
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing RiskMapService ---")

    # Initialize services (this assumes models are already trained and saved)
    kb = KnowledgeBaseService() # Loads sample data

    # Ensure models are trained before testing RiskMapService if they aren't saved
    # For PoC, we assume they were trained if __main__ of model scripts were run
    pd_m = PDModel()
    lgd_m = LGDModel()

    # Attempt to load models. If they don't exist, they need to be trained first.
    # This is a common pattern: try to load, if fails, then train.
    if not pd_m.load_model():
        logger.warning("PD model file not found. Training PD model for RiskMapService test...")
        if kb.get_all_loans() and kb.get_all_companies(): # Check if data is available
             pd_train_metrics = pd_m.train(kb_service=kb)
             logger.info(f"PD Model trained with metrics: {pd_train_metrics}")
             if "error" in pd_train_metrics:
                 logger.error("Failed to train PD model. RiskMapService tests may be incomplete.")
        else:
            logger.error("Cannot train PD model: No data in KB.")


    if not lgd_m.load_model():
        logger.warning("LGD model file not found. Training LGD model for RiskMapService test...")
        if kb.get_all_loans(): # Check if data is available
            lgd_train_metrics = lgd_m.train(kb_service=kb)
            logger.info(f"LGD Model trained with metrics: {lgd_train_metrics}")
            if "error" in lgd_train_metrics:
                 logger.error("Failed to train LGD model. RiskMapService tests may be incomplete.")
        else:
             logger.error("Cannot train LGD model: No data in KB.")


    # Initialize RiskMapService
    # Check if models were successfully loaded/trained before passing them
    if pd_m.model is None:
        logger.error("PD model is not available (failed to load/train). RiskMapService cannot function correctly.")
    if lgd_m.model is None:
        logger.error("LGD model is not available (failed to load/train). RiskMapService cannot function correctly.")

    # Proceed only if models are available
    if pd_m.model and lgd_m.model:
        # Initialize KnowledgeGraphService
        logger.info("Initializing KnowledgeGraphService for RiskMapService test...")
        kg_service = KnowledgeGraphService(kb_service=kb)
        # kg_service._populate_graph_from_kb() # Already called in KG __init__ if kb_service is provided

        risk_map_service = RiskMapService(kb_service=kb, pd_model=pd_m, lgd_model=lgd_m, kg_service=kg_service)

        # Generate portfolio overview
        portfolio_data = risk_map_service.generate_portfolio_risk_overview()
        if portfolio_data:
            logger.info(f"Generated {len(portfolio_data)} items for portfolio risk overview.")
            logger.info("First item in portfolio overview:")
            import json # Import json for pretty printing
            logger.info(json.dumps(portfolio_data[0], indent=2))
        else:
            logger.warning("Portfolio risk overview is empty.")

        # Generate summary by sector
        sector_summary_data = risk_map_service.get_risk_summary_by_sector(portfolio_overview=portfolio_data)
        if sector_summary_data:
            logger.info("Generated risk summary by sector:")
            logger.info(json.dumps(sector_summary_data, indent=2))
        else:
            logger.warning("Sector summary is empty.")

        # Generate summary by country
        country_summary_data = risk_map_service.get_risk_summary_by_country(portfolio_overview=portfolio_data)
        if country_summary_data:
            logger.info("Generated risk summary by country:")
            logger.info(json.dumps(country_summary_data, indent=2))
        else:
            logger.warning("Country summary is empty.")
    else:
        logger.error("RiskMapService could not be initialized due to missing models. Tests aborted.")
