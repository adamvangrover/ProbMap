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
        pd_shock_factor: float = 1.5,
        revenue_shock_factor: float = 0.8,
        scenario_name: str = "Economic Downturn"
    ) -> Dict[str, Any]:
        """
        Generates a simple economic shock scenario.
        This is a conceptual method. It would adjust features that PD/LGD models are sensitive to.

        Args:
            base_portfolio_df (pd.DataFrame): DataFrame of the current portfolio state.
                                             Expected to have columns like 'pd_estimate', 'company_revenue_usd_millions'.
            pd_shock_factor (float): Multiplier for existing PDs (e.g., 1.5 means PD increases by 50%).
            revenue_shock_factor (float): Multiplier for company revenues (e.g., 0.8 means 20% decrease).
            scenario_name (str): Name of the scenario.

        Returns:
            Dict[str, Any]: A dictionary containing the scenario name and the modified DataFrame.
        """
        if base_portfolio_df.empty:
            logger.warning("Base portfolio DataFrame is empty. Cannot generate shock scenario.")
            return {"name": scenario_name, "data": pd.DataFrame(), "description": "Empty base data."}

        logger.info(f"Generating '{scenario_name}' scenario with PD shock factor: {pd_shock_factor}, Revenue shock: {revenue_shock_factor}")

        shocked_df = base_portfolio_df.copy()

        # Apply PD shock (conceptual - actual PD model would be re-run with shocked inputs)
        # For this stub, we assume 'pd_estimate' is a column in base_portfolio_df
        if 'pd_estimate' in shocked_df.columns:
            shocked_df['original_pd_estimate'] = shocked_df['pd_estimate']
            shocked_df['pd_estimate'] = np.clip(shocked_df['pd_estimate'] * pd_shock_factor, 0, 1.0)
        else:
            logger.warning("'pd_estimate' column not found in base_portfolio_df. PD shock not applied directly.")

        # Apply revenue shock (this would be an input to PD/LGD models)
        # For this stub, we assume 'company_revenue_usd_millions' is a column
        revenue_col_options = ['company_revenue_usd_millions', 'revenue_usd_millions'] # Check for common names
        actual_revenue_col = None
        for col_name in revenue_col_options:
            if col_name in shocked_df.columns:
                actual_revenue_col = col_name
                break

        if actual_revenue_col:
            shocked_df[f'original_{actual_revenue_col}'] = shocked_df[actual_revenue_col]
            shocked_df[actual_revenue_col] = shocked_df[actual_revenue_col] * revenue_shock_factor
        else:
            logger.warning(f"Revenue column not found in {revenue_col_options}. Revenue shock not applied.")

        description = (f"Scenario: {scenario_name}. "
                       f"PDs shocked by factor {pd_shock_factor}. "
                       f"Revenues shocked by factor {revenue_shock_factor}.")

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
        'pd_estimate': [0.05, 0.10, 0.02],
        'company_revenue_usd_millions': [100, 50, 200]
    }
    sample_df = pd.DataFrame(sample_portfolio_data)

    economic_downturn_scenario = generator.generate_economic_shock_scenario(sample_df, pd_shock_factor=2.0, revenue_shock_factor=0.7)
    logger.info(f"Generated Scenario: {economic_downturn_scenario['name']}")
    logger.info(f"Description: {economic_downturn_scenario['description']}")
    if not economic_downturn_scenario['data'].empty:
        logger.info("Shocked Data (first few rows):")
        logger.info("\n" + economic_downturn_scenario['data'].head().to_string())
    else:
        logger.info("Shocked Data: DataFrame is empty.")

    # Test Monte Carlo input generation
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
