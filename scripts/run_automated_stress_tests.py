import sys
import os
import logging
import pandas as pd
import json
from pathlib import Path

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.simulation.scenario_generator import ScenarioGenerator
from src.simulation.stress_tester import StressTester
from src.data_management.knowledge_base import KnowledgeBaseService
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

def generate_report(results):
    output_path = PROJECT_ROOT / "output" / "stress_test_report_2026.md"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Automated Stress Test Report 2026\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now()}\n\n")

        f.write("| Scenario | Expected Loss (USD) | Avg PD | Avg LGD | Max Exposure |\n")
        f.write("|---|---|---|---|---|\n")

        for res in results:
            f.write(f"| {res['scenario']} | "
                    f"${res.get('total_expected_loss', 0):,.2f} | "
                    f"{res.get('average_pd', 0):.4f} | "
                    f"{res.get('average_lgd', 0):.4f} | "
                    f"${res.get('max_exposure', 0):,.2f} |\n")

        f.write("\n\n## Analysis\n")
        base = next((r for r in results if r['scenario'] == 'Base Case'), None)
        severe = next((r for r in results if r['scenario'] == 'Severe Global Crash'), None)

        if base and severe:
            el_increase = severe['total_expected_loss'] - base['total_expected_loss']
            pct_increase = (el_increase / base['total_expected_loss']) * 100 if base['total_expected_loss'] > 0 else 0
            f.write(f"Under a **Severe Global Crash**, Expected Loss increases by **${el_increase:,.2f}** (+{pct_increase:.1f}%).\n")

    logger.info(f"Report generated at: {output_path}")

def run_automated_stress_tests():
    logger.info("Starting Automated Stress Testing Pipeline...")

    # Initialize components
    kb_service = KnowledgeBaseService()

    pd_model = PDModel()
    if not pd_model.load_model():
        if kb_service.get_all_loans(): pd_model.train(kb_service)

    lgd_model = LGDModel()
    if not lgd_model.load_model():
        if kb_service.get_all_loans(): lgd_model.train(kb_service)

    if not pd_model.model or not lgd_model.model:
        logger.error("Models not ready. Aborting stress test.")
        return

    scenario_gen = ScenarioGenerator()
    stress_tester = StressTester(pd_model, lgd_model, kb_service)

    # Prepare Base Data (Portfolio)
    loans = kb_service.get_all_loans()
    companies = {c.company_id: c for c in kb_service.get_all_companies()}

    portfolio_data = []
    for loan in loans:
        comp = companies.get(loan.company_id)
        if not comp: continue

        item = {
            'loan_id': loan.loan_id,
            'company_id': loan.company_id,
            'loan_amount': loan.loan_amount,
            'interest_rate_percentage': loan.interest_rate_percentage,
            'company_revenue_usd_millions': comp.revenue_usd_millions,
            'industry_sector': comp.industry_sector.value if hasattr(comp.industry_sector, 'value') else str(comp.industry_sector),
            'economic_condition_indicator': loan.economic_condition_indicator or 0.5,
            # Add other necessary features for PD model
            'loan_duration_days': 365, # Mock
            'company_age_at_origination': 5, # Mock
            'debt_to_equity_ratio': 1.5, # Mock
            'current_ratio': 1.2, # Mock
            'net_profit_margin': 0.1, # Mock
            'roe': 0.15, # Mock
            'loan_amount_x_interest_rate': loan.loan_amount * (loan.interest_rate_percentage or 0),
            # LGD specific features
            'collateral_type': loan.collateral_type.value if hasattr(loan.collateral_type, 'value') else str(loan.collateral_type),
            'loan_amount_usd': loan.loan_amount # Duplicate for consistency
        }
        portfolio_data.append(item)

    base_df = pd.DataFrame(portfolio_data)
    if base_df.empty:
        logger.warning("Portfolio is empty. Cannot run tests.")
        return

    # Define Scenarios
    scenarios_config = [
        {
            "name": "Mild Recession 2026",
            "shocks": {
                "company_revenue_usd_millions": {"type": "multiplicative", "value": 0.90},
                "economic_condition_indicator": {"type": "override", "value": 0.4},
                "interest_rate_percentage": {"type": "additive", "value": 0.01}
            }
        },
        {
            "name": "Severe Global Crash",
            "shocks": {
                "company_revenue_usd_millions": {"type": "multiplicative", "value": 0.70},
                "economic_condition_indicator": {"type": "override", "value": 0.1},
                "interest_rate_percentage": {"type": "additive", "value": 0.03},
                "net_profit_margin": {"type": "additive", "value": -0.05} # Squeeze margins
            }
        },
        {
            "name": "Tech Sector Bubble Burst",
            # This requires filtering logic which ScenarioGenerator might genericize,
            # but for now we apply global shocks or need custom logic.
            # Simplified: Global tech revenue shock
            "shocks": {
                "company_revenue_usd_millions": {"type": "multiplicative", "value": 0.60}
            }
        }
    ]

    results = []

    # Helper to aggregate metrics from stress test output
    def process_stress_results(stress_output, scenario_name):
        details = stress_output.get("stressed_portfolio_detail", [])
        if not details:
            return {
                "scenario": scenario_name,
                "total_expected_loss": 0,
                "average_pd": 0,
                "average_lgd": 0,
                "max_exposure": 0
            }

        total_el = stress_output.get("total_stressed_expected_loss_usd", 0)
        pds = [d['stressed_pd_estimate'] for d in details if d['stressed_pd_estimate'] is not None]
        lgds = [d['stressed_lgd_estimate'] for d in details if d['stressed_lgd_estimate'] is not None]
        exposures = [d['exposure_at_default_usd'] for d in details]

        return {
            "scenario": scenario_name,
            "total_expected_loss": total_el,
            "average_pd": sum(pds) / len(pds) if pds else 0,
            "average_lgd": sum(lgds) / len(lgds) if lgds else 0,
            "max_exposure": max(exposures) if exposures else 0
        }

    # Run Base Case (as a scenario with no shocks)
    logger.info("Running Base Case...")
    base_scenario = {"name": "Base Case", "data": base_df}
    base_output = stress_tester.run_stress_test_on_portfolio(base_scenario)
    results.append(process_stress_results(base_output, "Base Case"))

    # Run Scenarios
    for scen_cfg in scenarios_config:
        logger.info(f"Running Scenario: {scen_cfg['name']}")
        scenario_output = scenario_gen.generate_economic_shock_scenario(
            base_df,
            scen_cfg['shocks'],
            scen_cfg['name']
        )

        stress_output = stress_tester.run_stress_test_on_portfolio(scenario_output)
        results.append(process_stress_results(stress_output, scen_cfg['name']))

    # Generate Markdown Report
    generate_report(results)

if __name__ == "__main__":
    run_automated_stress_tests()
