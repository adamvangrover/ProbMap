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
                        loan_data: Optional[Dict[str, Any]] = None,
                        kg_context: Optional[Dict[str, Any]] = None,
                        customer_segment: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates a suggested interest rate for a loan.

        Args:
            pd_estimate (float): Probability of Default (e.g., 0.05 for 5%).
            lgd_estimate (float): Loss Given Default (e.g., 0.40 for 40%).
            company_data (Optional[Dict[str, Any]]): Company features.
            loan_data (Optional[Dict[str, Any]]): Loan features.
            kg_context (Optional[Dict[str, Any]]): Contextual info from Knowledge Graph.
                                                   (e.g., {'degree_centrality': 0.1, 'num_suppliers': 5})
            customer_segment (Optional[str]): Customer segment (e.g., 'Strategic', 'Standard', 'HighRisk').
        Returns:
            Dict[str, Any]: A dictionary containing the suggested_interest_rate and other pricing details.
        """
        if not (0 <= pd_estimate <= 1):
            logger.warning(f"PD estimate {pd_estimate} is outside [0,1]. Clamping.")
            pd_estimate = max(0, min(1, pd_estimate))
        if not (0 <= lgd_estimate <= 1):
            logger.warning(f"LGD estimate {lgd_estimate} is outside [0,1]. Clamping.")
            lgd_estimate = max(0, min(1, lgd_estimate))

        # Dynamic Base Rate/Risk Premium Factor by Segment
        current_base_rate = self.base_rate
        current_risk_premium_factor = self.risk_premium_factor
        segment_applied = customer_segment if customer_segment else 'Standard' # Default to Standard

        if customer_segment == 'Strategic':
            current_base_rate *= 0.9
            current_risk_premium_factor *= 0.8
        elif customer_segment == 'HighRisk':
            current_base_rate *= 1.1
            current_risk_premium_factor *= 1.2

        # Basic risk premium calculation using potentially adjusted factors
        risk_premium = (pd_estimate * 100) * current_risk_premium_factor * (1 + (lgd_estimate * self.lgd_weight))

        # Industry sector adjustment
        industry_adjustment = 0.0
        if company_data and 'industry_sector' in company_data:
            sector = company_data['industry_sector']
            if sector == 'Technology': industry_adjustment = -0.25
            elif sector == 'Construction': industry_adjustment = 0.5

        # Loan amount adjustment
        loan_amount_adjustment = 0.0
        if loan_data and 'loan_amount_usd' in loan_data:
            if loan_data['loan_amount_usd'] > 5000000:
                loan_amount_adjustment = -0.1

        # KG Context Adjustment
        kg_adjustment = 0.0
        if kg_context:
            if kg_context.get('degree_centrality', 0.0) > 0.1: # Example threshold
                kg_adjustment -= 0.10 # Discount for high centrality

            num_suppliers = kg_context.get('num_suppliers', 5) # Default to a neutral number
            num_customers = kg_context.get('num_customers', 5) # Default to a neutral number
            if num_suppliers < 2 and num_customers < 2:
                kg_adjustment += 0.15 # Premium for low supplier/customer diversity

        suggested_rate = current_base_rate + risk_premium + industry_adjustment + loan_amount_adjustment + kg_adjustment

        # Ensure rate is not negative or excessively high
        suggested_rate = max(0.5, min(suggested_rate, 30.0))

        logger.info(f"Calculated price: PD={pd_estimate:.4f}, LGD={lgd_estimate:.4f}, Segment='{segment_applied}', KG_Adj={kg_adjustment:.2f} -> Suggested Rate={suggested_rate:.2f}%")

        return {
            "pd_input": pd_estimate,
            "lgd_input": lgd_estimate,
            "customer_segment_applied": segment_applied,
            "base_rate_component": self.base_rate, # Original base rate for reference
            "final_base_rate_used": current_base_rate,
            "risk_premium_factor_component": self.risk_premium_factor, # Original factor
            "final_risk_premium_factor_used": current_risk_premium_factor,
            "risk_premium_calculated_component": risk_premium,
            "industry_adjustment_component": industry_adjustment,
            "loan_amount_adjustment_component": loan_amount_adjustment,
            "kg_adjustment_component": kg_adjustment,
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
    logger.info(f"Scenario 1 (Low Risk, Standard Segment, No KG Context) Price: {price1['suggested_interest_rate']}%")
    logger.info(f"Details 1: {price1}")


    # Scenario 2: Medium risk, Strategic Segment
    pd2, lgd2 = 0.10, 0.50 # 10% PD, 50% LGD
    company2_data = {'industry_sector': 'Manufacturing', 'revenue_usd_millions': 50}
    loan2_data = {'loan_amount_usd': 2000000, 'duration_years': 5}
    price2 = pricing_model.calculate_price(pd2, lgd2, company2_data, loan2_data, customer_segment='Strategic')
    logger.info(f"Scenario 2 (Medium Risk, Strategic Segment) Price: {price2['suggested_interest_rate']}%")
    logger.info(f"Details 2: {price2}")

    # Scenario 3: High risk, HighRisk Segment
    pd3, lgd3 = 0.25, 0.75 # 25% PD, 75% LGD
    company3_data = {'industry_sector': 'Construction', 'revenue_usd_millions': 10}
    loan3_data = {'loan_amount_usd': 500000, 'duration_years': 7}
    price3 = pricing_model.calculate_price(pd3, lgd3, company3_data, loan3_data, customer_segment='HighRisk')
    logger.info(f"Scenario 3 (High Risk, HighRisk Segment) Price: {price3['suggested_interest_rate']}%")
    logger.info(f"Details 3: {price3}")

    # Scenario 4: With KG Context (High Centrality -> discount)
    pd4, lgd4 = 0.05, 0.40
    company4_data = {'industry_sector': 'Technology'}
    loan4_data = {'loan_amount_usd': 1000000}
    kg_context_good = {'degree_centrality': 0.25, 'num_suppliers': 5, 'num_customers': 10}
    price4 = pricing_model.calculate_price(pd4, lgd4, company4_data, loan4_data, kg_context=kg_context_good)
    logger.info(f"Scenario 4 (Low Risk, Good KG Context) Price: {price4['suggested_interest_rate']}%")
    logger.info(f"Details 4: {price4}")

    # Scenario 5: With KG Context (Low Diversity -> premium)
    pd5, lgd5 = 0.05, 0.40
    company5_data = {'industry_sector': 'Technology'}
    loan5_data = {'loan_amount_usd': 1000000}
    kg_context_risky = {'degree_centrality': 0.05, 'num_suppliers': 1, 'num_customers': 1}
    price5 = pricing_model.calculate_price(pd5, lgd5, company5_data, loan5_data, kg_context=kg_context_risky)
    logger.info(f"Scenario 5 (Low Risk, Risky KG Context) Price: {price5['suggested_interest_rate']}%")
    logger.info(f"Details 5: {price5}")

    # Scenario 6: Strategic segment with Risky KG context
    pd6, lgd6 = 0.05, 0.40
    company6_data = {'industry_sector': 'Technology'}
    loan6_data = {'loan_amount_usd': 1000000}
    price6 = pricing_model.calculate_price(pd6, lgd6, company6_data, loan6_data, kg_context=kg_context_risky, customer_segment='Strategic')
    logger.info(f"Scenario 6 (Low Risk, Strategic Segment, Risky KG Context) Price: {price6['suggested_interest_rate']}%")
    logger.info(f"Details 6: {price6}")
