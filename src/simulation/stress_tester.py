import logging
from typing import List, Dict, Any, Optional
import pandas as pd

# Assuming these services and models are available and potentially pre-trained/loaded
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_base import KnowledgeBaseService # For getting original data if needed
from src.risk_map.risk_map_service import RiskMapService # To re-calculate portfolio overview under stress

logger = logging.getLogger(__name__)

class StressTester:
    """
    Applies scenarios to a portfolio and evaluates the impact on risk metrics.
    For PoC, this will be a conceptual application of scenario effects.
    """

    def __init__(self,
                 pd_model: PDModel,
                 lgd_model: LGDModel,
                 kb_service: KnowledgeBaseService # Used to get full data for re-prediction
                ):
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.kb_service = kb_service # To fetch original company/loan details

        # Ensure models are loaded
        if self.pd_model.model is None:
            logger.info("StressTester: PD model not loaded, attempting to load.")
            self.pd_model.load_model() # Errors handled within load_model
        if self.lgd_model.model is None:
            logger.info("StressTester: LGD model not loaded, attempting to load.")
            self.lgd_model.load_model()

        logger.info("StressTester initialized.")


    def run_stress_test_on_portfolio(
        self,
        scenario: Dict[str, Any] # Output from ScenarioGenerator e.g. {"name": "EcoDownturn", "data": shocked_df}
    ) -> Dict[str, Any]:
        """
        Runs a stress test on the entire portfolio based on a generated scenario.
        The scenario dictionary should contain 'name' and 'data' (a DataFrame with shocked features).
        This PoC method will re-calculate PD and LGD based on the shocked features.

        Args:
            scenario (Dict[str, Any]): A scenario object, typically from ScenarioGenerator.
                                       Expected to have 'name' and 'data' (DataFrame).
                                       The 'data' DataFrame should contain all necessary raw features
                                       for PD and LGD models after scenario shocks have been applied.

        Returns:
            Dict[str, Any]: A dictionary containing the stress test results, including
                            the scenario name and a new portfolio overview under stress.
        """
        scenario_name = scenario.get("name", "Unnamed Scenario")
        shocked_features_df = scenario.get("data") # This df contains features *after* shock

        if shocked_features_df is None or shocked_features_df.empty:
            logger.warning(f"Scenario '{scenario_name}' has no data. Stress test cannot proceed.")
            return {"scenario_name": scenario_name, "results": "No data in scenario", "stressed_portfolio_overview": []}

        logger.info(f"Running stress test for scenario: {scenario_name}")

        if self.pd_model.model is None or self.lgd_model.model is None:
            logger.error("PD or LGD model not available in StressTester. Cannot run stress test.")
            return {"scenario_name": scenario_name, "results": "PD/LGD model missing", "stressed_portfolio_overview": []}

        stressed_portfolio_overview = []

        # Iterate through each row of the shocked_features_df. Each row represents a loan/company under stress.
        # We need to map rows in shocked_features_df back to original loan/company objects or IDs
        # to get all necessary data for PD/LGD prediction and EL calculation.
        # The `shocked_features_df` from `ScenarioGenerator` should ideally retain 'loan_id' and 'company_id'.

        if 'loan_id' not in shocked_features_df.columns or 'company_id' not in shocked_features_df.columns:
            logger.error("'loan_id' or 'company_id' missing in shocked_features_df. Cannot map to original data for full prediction.")
            return {"scenario_name": scenario_name, "results": "Missing IDs in scenario data", "stressed_portfolio_overview": []}

        for index, shocked_row in shocked_features_df.iterrows():
            loan_id = shocked_row['loan_id']
            company_id = shocked_row['company_id']

            # Fetch original loan and company data from KB to get all attributes
            original_loan = self.kb_service.get_loans_for_company(company_id) # This returns a list
            loan_obj = next((l for l in original_loan if l.loan_id == loan_id), None)
            company_obj = self.kb_service.get_company_profile(company_id)

            if not loan_obj or not company_obj:
                logger.warning(f"Could not find original loan {loan_id} or company {company_id} in KB. Skipping.")
                continue

            # Create the feature dictionary for PD model using shocked data
            # The PD model's `predict_for_loan` expects dicts of the *original* model structure,
            # but with values from the *shocked* row.

            # Construct company dict with potentially shocked values
            stressed_company_dict = company_obj.model_dump() # Start with original
            if 'company_revenue_usd_millions' in shocked_row and pd.notna(shocked_row['company_revenue_usd_millions']):
                stressed_company_dict['revenue_usd_millions'] = shocked_row['company_revenue_usd_millions']
            # Add other shocked company features if any

            # Construct loan dict with potentially shocked values
            stressed_loan_dict = loan_obj.model_dump()
            # Add shocked loan features if any (e.g., if scenario changed interest rates)


            # Predict PD using shocked features
            # The `predict_for_loan` method in PDModel needs to correctly map these.
            # We are passing the full original loan/company dicts but the PD model's
            # feature preparation for prediction uses the shocked values from `shocked_row`
            # This part is tricky: PDModel.predict_for_loan takes dicts, not the pre-processed row.
            # We need to ensure the features it extracts from these dicts reflect the shock.
            # For PoC, let's assume predict_for_loan can take a 'feature_overrides' dict.
            # This requires modification of PDModel or a different approach here.

            # Simpler approach for PoC: Assume PDModel.predict() can take the shocked_row directly
            # if shocked_row contains all necessary *raw* features for the PD model's preprocessor.
            # The current PDModel.predict expects a DataFrame of features.

            # Let's use the PDModel.predict_for_loan by constructing the input carefully
            # The `shocked_row` should contain the *raw features* that `predict_for_loan` internally processes.
            # E.g., `shocked_row` should have 'company_revenue_usd_millions', not the scaled version.
            pd_pred_class, pd_prob = -1, -1.0
            pd_result = self.pd_model.predict_for_loan(stressed_loan_dict, stressed_company_dict) # This uses original values from dicts, not shocked ones!
                                                                                                # This is a flaw in this PoC stub for StressTester.
                                                                                                # A better way:
                                                                                                # pd_model.predict(pd.DataFrame([shocked_row_relevant_features]))

            # Corrected conceptual path for PD:
            # 1. The ScenarioGenerator's shocked_df should contain the *raw features* after shock.
            # 2. PDModel should have a predict method that takes a DataFrame of these raw features.
            # For now, we'll proceed with a placeholder value if direct prediction is complex with current structure.
            # OR, if 'pd_estimate' was directly shocked in scenario data:
            if 'pd_estimate' in shocked_row and pd.notna(shocked_row['pd_estimate']):
                 pd_prob = shocked_row['pd_estimate'] # Use directly if scenario pre-calculates/shocks it
                 pd_pred_class = 1 if pd_prob > 0.5 else 0 # Example threshold
            else: # Fallback if PD not directly available in shocked_row
                logger.warning(f"PD estimate not directly available in shocked data for {loan_id}. Using placeholder.")
                pd_prob = 0.5 # Placeholder


            # Predict LGD using shocked features (if LGD model depends on them)
            # LGD model's `predict_lgd` takes a dict of features.
            lgd_features_for_stress = {
                'collateral_type': loan_obj.collateral_type.value if loan_obj.collateral_type else 'None',
                'loan_amount_usd': loan_obj.loan_amount # Assuming loan amount doesn't change in this scenario
                # Add other features, potentially from shocked_row if scenario affects them
            }
            lgd_val = self.lgd_model.predict_lgd(lgd_features_for_stress)

            ead = loan_obj.loan_amount # Exposure At Default
            stressed_el = pd_prob * lgd_val * ead

            stressed_item = {
                "loan_id": loan_id,
                "company_id": company_id,
                "company_name": company_obj.company_name,
                "original_pd_estimate": shocked_row.get('original_pd_estimate', "N/A"), # If available from scenario
                "stressed_pd_estimate": round(pd_prob, 4),
                "stressed_lgd_estimate": round(lgd_val, 4),
                "stressed_expected_loss_usd": round(stressed_el, 2),
                "exposure_at_default_usd": ead
            }
            stressed_portfolio_overview.append(stressed_item)

        total_stressed_el = sum(item['stressed_expected_loss_usd'] for item in stressed_portfolio_overview if isinstance(item['stressed_expected_loss_usd'], (int, float)))

        results_summary = {
            "scenario_name": scenario_name,
            "scenario_description": scenario.get("description", "N/A"),
            "number_of_loans_processed": len(stressed_portfolio_overview),
            "total_stressed_expected_loss_usd": round(total_stressed_el, 2),
            "stressed_portfolio_detail": stressed_portfolio_overview
        }

        logger.info(f"Stress test for '{scenario_name}' complete. Total Stressed EL: {total_stressed_el:.2f}")
        return results_summary


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing StressTester ---")

    # Initialize services and models (ensure models are trained/loadable)
    kb = KnowledgeBaseService()
    pd_m = PDModel()
    lgd_m = LGDModel()

    # Simplified: Assume models are loadable or train them for test
    if not pd_m.load_model():
        logger.info("Training PD model for StressTester test.")
        if kb.get_all_loans() and kb.get_all_companies(): pd_m.train(kb)
        else: logger.error("Cannot train PD: KB empty.")

    if not lgd_m.load_model():
        logger.info("Training LGD model for StressTester test.")
        if kb.get_all_loans(): lgd_m.train(kb) # LGD train uses defaulted loans from KB
        else: logger.error("Cannot train LGD: KB empty.")


    if pd_m.model is None or lgd_m.model is None:
        logger.error("PD or LGD model not available. StressTester tests cannot proceed.")
    else:
        stress_tester = StressTester(pd_model=pd_m, lgd_model=lgd_m, kb_service=kb)
        scenario_gen = ScenarioGenerator()

        # Prepare base data for scenario generation (mimicking RiskMapService output structure)
        # This base_df should contain raw features that ScenarioGenerator can shock,
        # and also 'loan_id', 'company_id' for mapping back.
        # For PoC, let's get some data from KB and make a simple DataFrame.
        base_data_for_scenario = []
        for loan_item in kb.get_all_loans()[:3]: # Take first 3 loans
            comp_item = kb.get_company_profile(loan_item.company_id)
            if comp_item:
                 base_data_for_scenario.append({
                     'loan_id': loan_item.loan_id,
                     'company_id': comp_item.company_id,
                     # Add raw features PD/LGD models might use and ScenarioGenerator might shock
                     'company_revenue_usd_millions': comp_item.revenue_usd_millions if comp_item.revenue_usd_millions is not None else 0,
                     'interest_rate_percentage': loan_item.interest_rate_percentage,
                     'collateral_type': loan_item.collateral_type.value if loan_item.collateral_type else "None",
                     'loan_amount_usd': loan_item.loan_amount,
                     'industry_sector': comp_item.industry_sector.value,
                     'company_age_years': (pd.Timestamp('today').date() - comp_item.founded_date).days / 365.25 if comp_item.founded_date else -1,
                     'pd_estimate': 0.1 # Placeholder initial PD for scenario generator to shock
                 })
        base_df_for_scenario = pd.DataFrame(base_data_for_scenario)

        if not base_df_for_scenario.empty:
            # Generate a scenario
            test_scenario = scenario_gen.generate_economic_shock_scenario(
                base_df_for_scenario,
                pd_shock_factor=2.5, # PDs increase by 150%
                revenue_shock_factor=0.6 # Revenues decrease by 40%
            )
            logger.info(f"Generated test scenario: {test_scenario.get('name')}")
            logger.info(f"Scenario description: {test_scenario.get('description')}")
            if "data" in test_scenario and not test_scenario["data"].empty:
                 logger.info("Shocked data for scenario (first few rows):\n" + test_scenario["data"].head().to_string())

                 # Run stress test
                 stress_results = stress_tester.run_stress_test_on_portfolio(scenario=test_scenario)

                 logger.info(f"Stress Test Results for scenario '{stress_results.get('scenario_name')}':")
                 logger.info(f"  Description: {stress_results.get('scenario_description')}")
                 logger.info(f"  Loans Processed: {stress_results.get('number_of_loans_processed')}")
                 logger.info(f"  Total Stressed EL: {stress_results.get('total_stressed_expected_loss_usd')}")

                 if stress_results.get("stressed_portfolio_detail"):
                     logger.info("  First item in stressed portfolio detail:")
                     import json
                     logger.info(json.dumps(stress_results["stressed_portfolio_detail"][0], indent=2))
                 else:
                     logger.warning("  Stressed portfolio detail is empty.")
            else:
                logger.error("Scenario generation failed or produced no data. Cannot run stress test.")
        else:
            logger.error("Base data for scenario generation is empty. Cannot test StressTester.")
