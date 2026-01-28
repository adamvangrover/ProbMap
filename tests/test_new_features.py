import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch

from src.risk_models.bayesian import BayesianCreditModel
from src.simulation.stress_tester import StressTester
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.knowledge_graph import KnowledgeGraphService
from src.data_management.ontology import LoanAgreement, CorporateEntity, IndustrySector, CollateralType

class TestNewFeatures(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.ERROR)

    def test_bayesian_credit_model_initialization(self):
        # Test default init
        model = BayesianCreditModel()
        probs = model.get_belief_distribution()
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        self.assertIn('AAA', probs)
        self.assertIn('D', probs)

        # Test custom init
        custom_priors = {'AAA': 0.5, 'D': 0.5} # Others should be 0 implicitly or handled?
        # The class expects provided keys. Let's see how it handles missing keys.
        # Actually it uses pd.Series(prior_probs). So others will be NaN if not provided?
        # Let's provide all for safety in this test or update class to handle partials.
        # The class assumes keys provided are the universe if not matching STATES.
        # But STATES are fixed.

        # Let's test with full keys for now as per likely usage
        full_custom = {s: (0.125) for s in BayesianCreditModel.STATES}
        model2 = BayesianCreditModel(full_custom)
        self.assertAlmostEqual(model2.get_belief_distribution()['AAA'], 0.125)

    def test_bayesian_update(self):
        model = BayesianCreditModel()
        initial_aaa = model.get_belief_distribution()['AAA']
        initial_d = model.get_belief_distribution()['D']

        # Apply positive evidence
        model.update('high_revenue_growth', strength='medium')
        new_aaa = model.get_belief_distribution()['AAA']
        new_d = model.get_belief_distribution()['D']

        self.assertTrue(new_aaa > initial_aaa, "AAA prob should increase with positive evidence")
        self.assertTrue(new_d < initial_d, "D prob should decrease with positive evidence")

        # Apply negative evidence
        model.update('lawsuit', strength='high')
        final_aaa = model.get_belief_distribution()['AAA']
        final_d = model.get_belief_distribution()['D']

        self.assertTrue(final_aaa < new_aaa, "AAA prob should decrease with negative evidence")
        self.assertTrue(final_d > new_d, "D prob should increase with negative evidence")

    def test_graph_stress_test(self):
        # Mock dependencies
        mock_kb = MagicMock(spec=KnowledgeBaseService)
        mock_pd = MagicMock(spec=PDModel)
        mock_lgd = MagicMock(spec=LGDModel)
        mock_kg = MagicMock(spec=KnowledgeGraphService)

        # Setup mock models to be "loaded"
        mock_pd.model = MagicMock()
        mock_lgd.model = MagicMock()

        # Setup mock graph
        # Mocking networkx graph
        mock_graph = MagicMock()
        mock_kg.graph = mock_graph

        # Define graph structure: C1 -> C2 -> C3
        # If C1 defaults, C2 gets shock, C3 gets smaller shock
        mock_graph.has_node.side_effect = lambda n: n in ['C1', 'C2', 'C3']

        def get_neighbors(node):
            if node == 'C1': return ['C2']
            if node == 'C2': return ['C3']
            return []

        mock_graph.neighbors.side_effect = get_neighbors

        # Setup KB data
        # We need loans linked to these companies
        mock_loan1 = MagicMock(spec=LoanAgreement)
        mock_loan1.loan_id = 'L1'
        mock_loan1.company_id = 'C1'
        mock_loan1.loan_amount = 1000
        mock_loan1.economic_condition_indicator = 0.5
        mock_loan1.interest_rate_percentage = 0.05
        mock_loan1.collateral_type = CollateralType.REAL_ESTATE
        mock_loan1.seniority_of_debt = "Senior Secured"
        mock_loan1.origination_date = "2020-01-01"
        mock_loan1.maturity_date = "2025-01-01"

        mock_loan2 = MagicMock(spec=LoanAgreement)
        mock_loan2.loan_id = 'L2'
        mock_loan2.company_id = 'C2'
        mock_loan2.loan_amount = 1000
        mock_loan2.economic_condition_indicator = 0.5
        mock_loan2.interest_rate_percentage = 0.05
        mock_loan2.collateral_type = CollateralType.REAL_ESTATE
        mock_loan2.seniority_of_debt = "Senior Secured"
        mock_loan2.origination_date = "2020-01-01"
        mock_loan2.maturity_date = "2025-01-01"

        mock_loan3 = MagicMock(spec=LoanAgreement)
        mock_loan3.loan_id = 'L3'
        mock_loan3.company_id = 'C3'
        mock_loan3.loan_amount = 1000
        mock_loan3.economic_condition_indicator = 0.5
        mock_loan3.interest_rate_percentage = 0.05
        mock_loan3.collateral_type = CollateralType.REAL_ESTATE
        mock_loan3.seniority_of_debt = "Senior Secured"
        mock_loan3.origination_date = "2020-01-01"
        mock_loan3.maturity_date = "2025-01-01"

        mock_kb.get_all_loans.return_value = [mock_loan1, mock_loan2, mock_loan3]

        # Setup company profiles
        def get_company(cid):
            c = MagicMock(spec=CorporateEntity)
            c.company_id = cid
            c.company_name = f"Comp {cid}"
            c.revenue_usd_millions = 100.0
            c.industry_sector = IndustrySector.TECHNOLOGY
            c.model_dump.return_value = {}
            return c

        mock_kb.get_company_profile.side_effect = get_company
        mock_kb.get_loans_for_company.side_effect = lambda cid: [l for l in [mock_loan1, mock_loan2, mock_loan3] if l.company_id == cid]

        # Mock predictions
        mock_pd.predict_for_loan.return_value = (0, 0.05) # Always 5% PD
        mock_lgd.predict_lgd.return_value = 0.4 # Always 40% LGD

        # Init StressTester
        tester = StressTester(pd_model=mock_pd, lgd_model=mock_lgd, kb_service=mock_kb, kg_service=mock_kg)

        # Run Graph Stress Test
        # Shock C1 (Default)
        results = tester.run_graph_stress_test(initial_shock_entities=['C1'], decay_factor=0.5)

        # Verify
        self.assertIn("Graph Contagion Simulation", results['scenario_name'])

        # Check details
        details = results['stressed_portfolio_detail']

        # Find entries for C1, C2, C3
        c1_res = next(d for d in details if d['company_id'] == 'C1')
        c2_res = next(d for d in details if d['company_id'] == 'C2')
        c3_res = next(d for d in details if d['company_id'] == 'C3')

        # C1 should have revenue reduced to 0 (Shock 1.0)
        self.assertEqual(c1_res['shocked_features']['company_revenue_usd_millions'], 0.0)

        # C2 should have revenue reduced by 50% (Shock 0.5)
        self.assertEqual(c2_res['shocked_features']['company_revenue_usd_millions'], 50.0)

        # C3 should have revenue reduced by 25% (Shock 0.25)
        self.assertEqual(c3_res['shocked_features']['company_revenue_usd_millions'], 75.0)

if __name__ == '__main__':
    unittest.main()
