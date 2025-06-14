import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class PricingModel:
    """
    Credit Pricing Model.
    For PoC, this uses a simple formula based on PD, LGD, and other factors.
    """
    def __init__(self, base_rate: float = 3.0, risk_premium_factor: float = 0.5, lgd_weight: float = 0.7):
        self.base_rate = base_rate  # Base interest rate percentage
        self.risk_premium_factor = risk_premium_factor # Multiplier for PD to add to base rate
        self.lgd_weight = lgd_weight # How much LGD influences the premium (0 to 1)
        logger.info(f"PricingModel initialized with base_rate={base_rate}, risk_premium_factor={risk_premium_factor}, lgd_weight={lgd_weight}")

    def calculate_price(self, pd_estimate: float, lgd_estimate: float,
                        company_data: Optional[Dict[str, Any]] = None,
                        loan_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Calculates a suggested interest rate for a loan.

        Args:
            pd_estimate (float): Probability of Default (e.g., 0.05 for 5%).
            lgd_estimate (float): Loss Given Default (e.g., 0.40 for 40%).
            company_data (Optional[Dict[str, Any]]): Dictionary of company features.
                                                     (e.g., {'industry_sector': 'Technology', 'revenue_usd_millions': 100})
            loan_data (Optional[Dict[str, Any]]): Dictionary of loan features.
                                                  (e.g., {'loan_amount_usd': 1000000, 'duration_years': 5})
        Returns:
            Dict[str, Any]: A dictionary containing the suggested_interest_rate and other pricing details.
        """
        if not (0 <= pd_estimate <= 1):
            logger.warning(f"PD estimate {pd_estimate} is outside [0,1]. Clamping.")
            pd_estimate = max(0, min(1, pd_estimate))
        if not (0 <= lgd_estimate <= 1):
            logger.warning(f"LGD estimate {lgd_estimate} is outside [0,1]. Clamping.")
            lgd_estimate = max(0, min(1, lgd_estimate))

        # Basic risk premium calculation
        # Risk premium increases with PD and is amplified by LGD
        # This is a very simplified formula for PoC.
        risk_premium = (pd_estimate * 100) * self.risk_premium_factor * (1 + (lgd_estimate * self.lgd_weight))

        # Industry sector adjustment (example qualitative factor)
        industry_adjustment = 0.0
        if company_data and 'industry_sector' in company_data:
            sector = company_data['industry_sector']
            if sector == 'Technology': industry_adjustment = -0.25 # Lower rate for perceived lower risk sector
            elif sector == 'Construction': industry_adjustment = 0.5 # Higher rate

        # Loan amount/duration adjustment (example)
        loan_amount_adjustment = 0.0
        if loan_data and 'loan_amount_usd' in loan_data:
            if loan_data['loan_amount_usd'] > 5000000: # Larger loans might get slightly better rates (or worse, depends on policy)
                loan_amount_adjustment = -0.1

        suggested_rate = self.base_rate + risk_premium + industry_adjustment + loan_amount_adjustment

        # Ensure rate is not negative or excessively high
        suggested_rate = max(0.5, min(suggested_rate, 30.0)) # Floor at 0.5%, cap at 30%

        logger.info(f"Calculated price: PD={pd_estimate:.4f}, LGD={lgd_estimate:.4f} -> Suggested Rate={suggested_rate:.2f}%")

        return {
            "pd_input": pd_estimate,
            "lgd_input": lgd_estimate,
            "base_rate_component": self.base_rate,
            "risk_premium_component": risk_premium,
            "industry_adjustment_component": industry_adjustment,
            "loan_amount_adjustment_component": loan_amount_adjustment,
            "suggested_interest_rate": round(suggested_rate, 2)
        }

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing PricingModel ---")
    pricing_model = PricingModel()

    # Scenario 1: Low risk
    pd1, lgd1 = 0.02, 0.30 # 2% PD, 30% LGD
    company1_data = {'industry_sector': 'Technology', 'revenue_usd_millions': 200}
    loan1_data = {'loan_amount_usd': 1000000, 'duration_years': 3}
    price1 = pricing_model.calculate_price(pd1, lgd1, company1_data, loan1_data)
    logger.info(f"Scenario 1 (Low Risk) Price: {price1['suggested_interest_rate']}%")
    # logger.info(f"Details: {price1}")


    # Scenario 2: Medium risk
    pd2, lgd2 = 0.10, 0.50 # 10% PD, 50% LGD
    company2_data = {'industry_sector': 'Manufacturing', 'revenue_usd_millions': 50}
    loan2_data = {'loan_amount_usd': 2000000, 'duration_years': 5}
    price2 = pricing_model.calculate_price(pd2, lgd2, company2_data, loan2_data)
    logger.info(f"Scenario 2 (Medium Risk) Price: {price2['suggested_interest_rate']}%")

    # Scenario 3: High risk
    pd3, lgd3 = 0.25, 0.75 # 25% PD, 75% LGD
    company3_data = {'industry_sector': 'Construction', 'revenue_usd_millions': 10}
    loan3_data = {'loan_amount_usd': 500000, 'duration_years': 7}
    price3 = pricing_model.calculate_price(pd3, lgd3, company3_data, loan3_data)
    logger.info(f"Scenario 3 (High Risk) Price: {price3['suggested_interest_rate']}%")

    # Scenario 4: Edge case (very high PD)
    pd4, lgd4 = 0.80, 0.90
    price4 = pricing_model.calculate_price(pd4, lgd4)
    logger.info(f"Scenario 4 (Very High Risk) Price: {price4['suggested_interest_rate']}%")
