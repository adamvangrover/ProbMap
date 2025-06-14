import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ScenarioGenerator:
    """
    Generates hypothetical scenarios for stress testing.
    For PoC, this will create very simple variations of input data.
    A real system would involve complex macroeconomic models or user-defined shocks.
    """

    def __init__(self):
        logger.info("ScenarioGenerator initialized.")

    def generate_economic_shock_scenario(
        self,
        base_portfolio_df: pd.DataFrame,
        feature_shocks: Dict[str, Dict[str, Any]],
        scenario_name: str = "Custom Economic Shock"
    ) -> Dict[str, Any]:
        """
        Generates an economic shock scenario by applying specified shocks to features.

        Args:
            base_portfolio_df (pd.DataFrame): DataFrame of the current portfolio state.
                                             Expected to have columns corresponding to feature names in feature_shocks.
            feature_shocks (Dict[str, Dict[str, Any]]): Dictionary mapping feature names to shock details.
                Example: {'company_revenue_usd_millions': {'type': 'multiplicative', 'value': 0.8},
                          'interest_rate_percentage': {'type': 'additive', 'value': 0.01}}
            scenario_name (str): Name of the scenario.

        Returns:
            Dict[str, Any]: A dictionary containing the scenario name, the modified DataFrame, and a description.
        """
        if base_portfolio_df.empty:
            logger.warning("Base portfolio DataFrame is empty. Cannot generate shock scenario.")
            return {"name": scenario_name, "data": pd.DataFrame(), "description": "Empty base data."}

        logger.info(f"Generating '{scenario_name}' scenario with feature shocks: {feature_shocks}")

        shocked_df = base_portfolio_df.copy()
        applied_shocks_desc = []

        for feature_name, shock_details in feature_shocks.items():
            if feature_name in shocked_df.columns:
                shock_type = shock_details.get('type')
                shock_value = shock_details.get('value')

                # Store original value before applying shock
                shocked_df[f'original_{feature_name}'] = shocked_df[feature_name]

                if shock_type == 'multiplicative':
                    shocked_df[feature_name] = shocked_df[feature_name] * shock_value
                    applied_shocks_desc.append(f"{feature_name} multiplied by {shock_value}")
                elif shock_type == 'additive':
                    shocked_df[feature_name] = shocked_df[feature_name] + shock_value
                    applied_shocks_desc.append(f"{feature_name} increased by {shock_value}")
                elif shock_type == 'override':
                    shocked_df[feature_name] = shock_value
                    applied_shocks_desc.append(f"{feature_name} overridden to {shock_value}")
                else:
                    logger.warning(f"Unknown shock type '{shock_type}' for feature '{feature_name}'. Shock not applied.")
                    continue

                # Example: Simple clipping for revenue to avoid negative values if not desired
                if 'revenue' in feature_name.lower() and shock_type == 'multiplicative':
                    shocked_df[feature_name] = np.maximum(0, shocked_df[feature_name]) # Ensure revenue >= 0
                elif 'revenue' in feature_name.lower() and shock_type == 'additive' and shock_value < 0 :
                     shocked_df[feature_name] = np.maximum(0, shocked_df[feature_name]) # Ensure revenue >= 0
                # Example: Clip interest rates if needed (e.g., to be non-negative)
                if 'interest_rate' in feature_name.lower():
                     shocked_df[feature_name] = np.maximum(0.001, shocked_df[feature_name]) # Min 0.1%

            else:
                logger.warning(f"Feature '{feature_name}' specified in feature_shocks not found in DataFrame. Shock not applied.")

        # Remove direct PD shock logic
        # if 'pd_estimate' in shocked_df.columns:
        #     if 'original_pd_estimate' not in shocked_df.columns: # Store original only if not already stored by a direct shock on pd_estimate
        #          shocked_df['original_pd_estimate'] = shocked_df['pd_estimate']
        #     # PD will be re-calculated by StressTester based on shocked features.
        #     # No direct shock to 'pd_estimate' here.

        description = f"Scenario: {scenario_name}. Applied shocks: {'; '.join(applied_shocks_desc)}."
        if not applied_shocks_desc:
            description = f"Scenario: {scenario_name}. No shocks were applied (e.g. features not found or no shocks defined)."


        return {"name": scenario_name, "data": shocked_df, "description": description}

    def generate_monte_carlo_inputs(self, base_input_features: Dict[str, Any], num_simulations: int = 1000) -> List[Dict[str, Any]]:
        """
        Generates inputs for Monte Carlo simulation by varying base features.
        This is highly conceptual for PoC.

        Args:
            base_input_features (Dict[str, Any]): A dictionary of features for a single entity or loan.
                                                 e.g., {'revenue': 100, 'interest_rate': 0.05}
            num_simulations (int): Number of simulation variants to generate.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a simulated set of inputs.
        """
        logger.info(f"Generating {num_simulations} Monte Carlo input sets based on: {base_input_features}")
        simulated_inputs = []

        # Example: Vary revenue and interest rate using a normal distribution
        # This would require defining distributions and correlations for each feature in a real system.
        for _ in range(num_simulations):
            sim_input = base_input_features.copy()
            if 'company_revenue_usd_millions' in sim_input:
                mean_rev = sim_input['company_revenue_usd_millions']
                std_dev_rev = mean_rev * 0.2 # Assume 20% std dev for revenue
                sim_input['company_revenue_usd_millions'] = np.random.normal(mean_rev, std_dev_rev)

            if 'interest_rate_percentage' in sim_input:
                mean_rate = sim_input['interest_rate_percentage']
                std_dev_rate = mean_rate * 0.1 # Assume 10% std dev for interest rate
                sim_input['interest_rate_percentage'] = np.random.normal(mean_rate, std_dev_rate)

            # Add more feature variations as needed
            simulated_inputs.append(sim_input)

        return simulated_inputs


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing ScenarioGenerator ---")
    generator = ScenarioGenerator()

    # Test economic shock scenario
    sample_portfolio_data = {
        'loan_id': ['L001', 'L002', 'L003'],
        'company_id': ['C001', 'C002', 'C003'],
        'pd_estimate': [0.05, 0.10, 0.02], # This will be ignored by new method but kept for df structure
        'company_revenue_usd_millions': [100.0, 50.0, 200.0],
        'interest_rate_percentage': [0.05, 0.06, 0.045], # Added for potential shock
        'economic_condition_indicator': [0.6, 0.5, 0.7] # Added for potential shock
    }
    sample_df = pd.DataFrame(sample_portfolio_data)

    feature_shocks_config = {
        'company_revenue_usd_millions': {'type': 'multiplicative', 'value': 0.7}, # 30% decrease
        'interest_rate_percentage': {'type': 'additive', 'value': 0.02}, # 2% (200bps) absolute increase
        'economic_condition_indicator': {'type': 'override', 'value': 0.25} # Override to a poor state
    }

    economic_downturn_scenario = generator.generate_economic_shock_scenario(
        sample_df,
        feature_shocks=feature_shocks_config,
        scenario_name="Severe Economic Downturn"
    )
    logger.info(f"Generated Scenario: {economic_downturn_scenario['name']}")
    logger.info(f"Description: {economic_downturn_scenario['description']}")
    if not economic_downturn_scenario['data'].empty:
        logger.info("Shocked Data (first few rows):")
        # Display relevant original and shocked columns for verification
        cols_to_show = ['loan_id', 'company_id']
        for col in feature_shocks_config.keys():
            cols_to_show.append(f'original_{col}')
            cols_to_show.append(col)
        # Ensure all columns exist in the dataframe before trying to display
        cols_to_show_existing = [col for col in cols_to_show if col in economic_downturn_scenario['data'].columns]
        logger.info("\n" + economic_downturn_scenario['data'][cols_to_show_existing].head().to_string())
    else:
        logger.info("Shocked Data: DataFrame is empty.")

    # Test Monte Carlo input generation (remains the same conceptually)
    sample_loan_features = {
        'company_revenue_usd_millions': 150.0,
        'interest_rate_percentage': 5.5,
        'loan_amount_usd': 5000000
        # other features needed by PD/LGD models...
    }
    mc_inputs = generator.generate_monte_carlo_inputs(sample_loan_features, num_simulations=5)
    logger.info(f"Generated {len(mc_inputs)} Monte Carlo input sets. First set:")
    if mc_inputs:
        import json
        logger.info(json.dumps(mc_inputs[0], indent=2))
