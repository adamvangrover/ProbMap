import logging
from typing import List, Dict, Any, Optional
import pandas as pd

# Assuming these services and models are available and potentially pre-trained/loaded
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_base import KnowledgeBaseService # For getting original data if needed
# from src.risk_map.risk_map_service import RiskMapService # Not directly used here

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
            # Construct company dict with potentially shocked values for PD model
            company_features_for_pd = company_obj.model_dump()
            # Override with shocked values if they exist in shocked_row
            for col, val in shocked_row.items():
                if col in company_features_for_pd and pd.notna(val):
                    company_features_for_pd[col] = val

            # Construct loan dict with potentially shocked values for PD model
            loan_features_for_pd = loan_obj.model_dump()
            for col, val in shocked_row.items():
                if col in loan_features_for_pd and pd.notna(val):
                    loan_features_for_pd[col] = val

            pd_pred_class, pd_prob = -1, 0.5 # Default values
            pd_result = self.pd_model.predict_for_loan(loan_features_for_pd, company_features_for_pd)
            if pd_result:
                pd_pred_class, pd_prob = pd_result
            else:
                logger.warning(f"Could not re-calculate PD for loan {loan_id} under stress. Using default 0.5.")

            # Construct features for LGD model using shocked data
            lgd_features_for_stress = {
                'collateral_type': str(shocked_row.get('collateral_type', loan_obj.collateral_type.value if loan_obj.collateral_type else 'None')),
                'loan_amount_usd': float(shocked_row.get('loan_amount_usd', loan_obj.loan_amount)),
                'seniority_of_debt': str(shocked_row.get('seniority_of_debt', loan_obj.seniority_of_debt if hasattr(loan_obj, 'seniority_of_debt') and loan_obj.seniority_of_debt else 'Unknown')),
                'economic_condition_indicator': float(shocked_row.get('economic_condition_indicator', loan_obj.economic_condition_indicator if hasattr(loan_obj, 'economic_condition_indicator') and loan_obj.economic_condition_indicator is not None else 0.5))
            }
            lgd_val = self.lgd_model.predict_lgd(lgd_features_for_stress)

            ead = float(shocked_row.get('loan_amount_usd', loan_obj.loan_amount)) # EAD might be the shocked loan amount if applicable
            stressed_el = pd_prob * lgd_val * ead

            # Determine original PD for comparison.
            # This requires predict_for_loan with unshocked data or access to an original_pd_estimate column.
            # If 'original_pd_estimate' is reliably passed through by ScenarioGenerator, use that.
            # Otherwise, it might be better to recalculate or set to N/A.
            original_pd_to_log = shocked_row.get('original_pd_estimate', "N/A")
            if original_pd_to_log == "N/A" and 'pd_estimate' in base_portfolio_df.columns: # if base_portfolio_df was accessible and had it
                 original_pd_to_log = base_portfolio_df.loc[index, 'pd_estimate'] # This is conceptual as base_portfolio_df isn't directly here.


            stressed_item = {
                "loan_id": loan_id,
                "company_id": company_id,
                "company_name": company_obj.company_name,
                # "original_pd_estimate": original_pd_to_log, # Keep if available and meaningful
                "original_features": {f"original_{k}": v for k, v in shocked_row.items() if k.startswith('original_')},
                "shocked_features": {k: v for k, v in shocked_row.items() if not k.startswith('original_') and k not in ['loan_id', 'company_id']},
                "stressed_pd_estimate": round(pd_prob, 4),
                "stressed_lgd_estimate": round(lgd_val, 4),
                "stressed_expected_loss_usd": round(stressed_el, 2) if pd.notna(stressed_el) else "N/A",
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
                     # Raw features for PDModel's _prepare_features -> predict_for_loan
                     'company_revenue_usd_millions': comp_item.revenue_usd_millions if comp_item.revenue_usd_millions is not None else 0,
                     'interest_rate_percentage': loan_item.interest_rate_percentage,
                     'collateral_type': loan_item.collateral_type.value if loan_item.collateral_type else "None",
                     'loan_amount_usd': loan_item.loan_amount,
                     'industry_sector': comp_item.industry_sector.value if comp_item.industry_sector else "Other",
                     'founded_date': str(comp_item.founded_date) if comp_item.founded_date else None, # For company_age_at_origination
                     'origination_date': str(loan_item.origination_date), # For company_age_at_origination & loan_duration
                     'maturity_date': str(loan_item.maturity_date), # For loan_duration
                     # Raw features for LGDModel's predict_lgd
                     'seniority_of_debt': str(loan_item.seniority_of_debt) if hasattr(loan_item, 'seniority_of_debt') and loan_item.seniority_of_debt else 'Unknown',
                     'economic_condition_indicator': loan_item.economic_condition_indicator if hasattr(loan_item, 'economic_condition_indicator') and loan_item.economic_condition_indicator is not None else 0.5,
                     # Original PD (optional, if you want to compare against a non-stressed version for logging)
                     # 'original_pd_estimate': # Could be from a non-stressed model run or a static value
                 })
        base_df_for_scenario = pd.DataFrame(base_data_for_scenario)

        if not base_df_for_scenario.empty:
            # Generate a scenario using feature_shocks
            feature_shocks_config = {
                'company_revenue_usd_millions': {'type': 'multiplicative', 'value': 0.6}, # 40% decrease
                'interest_rate_percentage': {'type': 'additive', 'value': 0.03},       # 3% (300bps) absolute increase
                'economic_condition_indicator': {'type': 'override', 'value': 0.15}    # Severe downturn
            }
            test_scenario = scenario_gen.generate_economic_shock_scenario(
                base_df_for_scenario,
                feature_shocks=feature_shocks_config,
                scenario_name="Severe Market Contraction"
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
