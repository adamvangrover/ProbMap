import unittest
from unittest.mock import MagicMock, PropertyMock
import datetime
import networkx as nx # Added import

from src.risk_map.risk_map_service import RiskMapService
from src.data_management.knowledge_base import KnowledgeBaseService
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_graph import KnowledgeGraphService, RelationshipType
from src.data_management.ontology import (
    LoanAgreement, CorporateEntity, IndustrySector, Currency, CollateralType # Removed SeniorityOfDebt
)

class TestRiskMapService(unittest.TestCase):

    def setUp(self):
        self.mock_kb_service = MagicMock(spec=KnowledgeBaseService)
        self.mock_pd_model = MagicMock(spec=PDModel)
        self.mock_lgd_model = MagicMock(spec=LGDModel)
        self.mock_kg_service = MagicMock(spec=KnowledgeGraphService)
        self.mock_kg_service.graph = MagicMock(spec=nx.MultiDiGraph) # Ensure graph attribute is a mock

        # Ensure models are "loaded" for RiskMapService __init__ checks
        type(self.mock_pd_model).model = PropertyMock(return_value=True) # Mock that model object exists
        type(self.mock_lgd_model).model = PropertyMock(return_value=True)


        self.risk_map_service = RiskMapService(
            kb_service=self.mock_kb_service,
            pd_model=self.mock_pd_model,
            lgd_model=self.mock_lgd_model,
            kg_service=self.mock_kg_service
        )

        # Sample data
        self.company1 = CorporateEntity(
            company_id="C1", company_name="Corp Alpha", industry_sector=IndustrySector.TECHNOLOGY,
            country_iso_code="US", founded_date=datetime.date(2000,1,1), management_quality_score=8
        )
        self.loan1 = LoanAgreement(
            loan_id="L1", company_id="C1", loan_amount=100000, currency=Currency.USD, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2022,1,1), maturity_date=datetime.date(2025,1,1),
            interest_rate_percentage=5.0, collateral_type=CollateralType.REAL_ESTATE,
            seniority_of_debt="Senior", default_status=False,
            economic_condition_indicator=0.6
        )
        self.company2 = CorporateEntity(
            company_id="C2", company_name="Corp Beta", industry_sector=IndustrySector.FINANCIAL_SERVICES, # Changed to FINANCIAL_SERVICES
            country_iso_code="GB", founded_date=datetime.date(2005,1,1), management_quality_score=6
        )
        self.loan2 = LoanAgreement(
            loan_id="L2", company_id="C2", loan_amount=50000, currency=Currency.GBP, # Changed CurrencyCode to Currency
            origination_date=datetime.date(2023,1,1), maturity_date=datetime.date(2024,1,1),
            interest_rate_percentage=7.0, collateral_type=CollateralType.NONE,
            seniority_of_debt="Junior", default_status=True, # Defaulted loan
            economic_condition_indicator=0.3
        )


    def test_generate_portfolio_risk_overview_successful(self):
        """Test successful generation of portfolio risk overview."""
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2]
        def get_company_profile_side_effect(company_id):
            if company_id == "C1": return self.company1
            if company_id == "C2": return self.company2
            return None
        self.mock_kb_service.get_company_profile.side_effect = get_company_profile_side_effect

        self.mock_pd_model.predict_for_loan.side_effect = [
            (0, 0.1), # PD for L1 (class, probability)
            (1, 0.6)  # PD for L2
        ]
        self.mock_lgd_model.predict_lgd.side_effect = [
            0.4, # LGD for L1
            0.8  # LGD for L2
        ]

        # Mock KG Service responses
        self.mock_kg_service.graph.has_node.return_value = True # Assume nodes exist in KG
        self.mock_kg_service.get_company_contextual_info.side_effect = [
            {"degree_centrality": 0.5, "num_suppliers": 5, "num_customers": 10, "num_subsidiaries": 2}, # For C1
            {"degree_centrality": 0.3, "num_suppliers": 2, "num_customers": 3, "num_subsidiaries": 0}  # For C2
        ]

        overview = self.risk_map_service.generate_portfolio_risk_overview()

        self.assertEqual(len(overview), 2)

        # Check item 1 (Loan L1, Company C1)
        item1 = next(item for item in overview if item["loan_id"] == "L1")
        self.assertEqual(item1["company_id"], "C1")
        self.assertEqual(item1["pd_estimate"], 0.1)
        self.assertEqual(item1["lgd_estimate"], 0.4)
        self.assertEqual(item1["exposure_at_default_usd"], self.loan1.loan_amount)
        expected_el1 = 0.1 * 0.4 * self.loan1.loan_amount
        self.assertAlmostEqual(item1["expected_loss_usd"], round(expected_el1, 2))
        self.assertEqual(item1["industry_sector"], IndustrySector.TECHNOLOGY.value)
        self.assertEqual(item1["country_iso_code"], "US")
        self.assertEqual(item1["kg_degree_centrality"], 0.5)
        self.assertEqual(item1["kg_num_suppliers"], 5)
        self.assertFalse(item1["is_defaulted"])


        # Check item 2 (Loan L2, Company C2)
        item2 = next(item for item in overview if item["loan_id"] == "L2")
        self.assertEqual(item2["company_id"], "C2")
        self.assertEqual(item2["pd_estimate"], 0.6)
        self.assertEqual(item2["lgd_estimate"], 0.8)
        expected_el2 = 0.6 * 0.8 * self.loan2.loan_amount
        self.assertAlmostEqual(item2["expected_loss_usd"], round(expected_el2, 2))
        self.assertEqual(item2["kg_num_customers"], 3)
        self.assertTrue(item2["is_defaulted"])


    def test_generate_overview_pd_prediction_fails(self):
        """Test overview generation when PD prediction returns None."""
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = None # PD prediction fails
        self.mock_lgd_model.predict_lgd.return_value = 0.4
        self.mock_kg_service.graph.has_node.return_value = False


        overview = self.risk_map_service.generate_portfolio_risk_overview()
        item1 = overview[0]
        self.assertEqual(item1["pd_estimate"], 0.5) # Default PD
        # EL should be calculated with default PD or be "N/A" if default PD is negative
        # Current default PD is 0.5, default LGD is 0.75 if model fails.
        # Here LGD model works (0.4). So EL = 0.5 * 0.4 * amount
        expected_el = 0.5 * 0.4 * self.loan1.loan_amount
        self.assertAlmostEqual(item1["expected_loss_usd"], round(expected_el,2))


    def test_generate_overview_lgd_model_unavailable(self):
        """Test overview generation when LGD model is not 'loaded' (model attribute is None)."""
        type(self.mock_lgd_model).model = PropertyMock(return_value=None) # Simulate LGD model not loaded
        # Re-init service to pick up the change in mock_lgd_model property
        risk_map_service_no_lgd = RiskMapService(
            self.mock_kb_service, self.mock_pd_model, self.mock_lgd_model, self.mock_kg_service
        )

        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1)
        self.mock_kg_service.graph.has_node.return_value = False # Mock for the new service instance


        overview = risk_map_service_no_lgd.generate_portfolio_risk_overview()
        item1 = overview[0]
        self.assertEqual(item1["lgd_estimate"], 0.75) # Default LGD
        expected_el = 0.1 * 0.75 * self.loan1.loan_amount
        self.assertAlmostEqual(item1["expected_loss_usd"], round(expected_el,2))


    def test_generate_overview_company_not_found(self):
        """Test when a company for a loan is not found in KB."""
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = None # Company not found
        overview = self.risk_map_service.generate_portfolio_risk_overview()
        self.assertEqual(len(overview), 0) # Loan should be skipped


    def test_generate_overview_no_kg_service(self):
        """Test when KG service is not provided."""
        risk_map_service_no_kg = RiskMapService(
            self.mock_kb_service, self.mock_pd_model, self.mock_lgd_model, kg_service=None
        )
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1)
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = risk_map_service_no_kg.generate_portfolio_risk_overview()
        item1 = overview[0]
        self.assertEqual(item1["kg_degree_centrality"], "N/A")
        self.assertEqual(item1["kg_num_suppliers"], "N/A")


    def test_get_risk_summary_by_sector(self):
        """Test aggregation of risk by industry sector."""
        # Sample portfolio overview
        portfolio_overview = [
            {"loan_id": "L1", "company_id": "C1", "industry_sector": "TECHNOLOGY", "country_iso_code": "US",
             "exposure_at_default_usd": 100000, "pd_estimate": 0.1, "lgd_estimate": 0.4,
             "expected_loss_usd": 4000.00, "is_defaulted": False}, # EL = 0.1*0.4*100k = 4000
            {"loan_id": "L2", "company_id": "C2", "industry_sector": "FINANCE", "country_iso_code": "GB",
             "exposure_at_default_usd": 50000, "pd_estimate": 0.6, "lgd_estimate": 0.8,
             "expected_loss_usd": 24000.00, "is_defaulted": True}, # EL = 0.6*0.8*50k = 24000
            {"loan_id": "L3", "company_id": "C3", "industry_sector": "TECHNOLOGY", "country_iso_code": "US",
             "exposure_at_default_usd": 200000, "pd_estimate": 0.05, "lgd_estimate": 0.5,
             "expected_loss_usd": 5000.00, "is_defaulted": False}  # EL = 0.05*0.5*200k = 5000
        ]

        summary = self.risk_map_service.get_risk_summary_by_sector(portfolio_overview)

        self.assertIn("TECHNOLOGY", summary)
        self.assertIn("FINANCE", summary)

        tech_summary = summary["TECHNOLOGY"]
        self.assertEqual(tech_summary["loan_count"], 2)
        self.assertEqual(tech_summary["total_exposure"], 100000 + 200000)
        self.assertAlmostEqual(tech_summary["total_expected_loss"], 4000.00 + 5000.00)
        self.assertAlmostEqual(tech_summary["average_pd"], (0.1 + 0.05) / 2)
        self.assertAlmostEqual(tech_summary["average_lgd"], (0.4 + 0.5) / 2)
        self.assertEqual(tech_summary["defaulted_loan_count"],0)


        fin_summary = summary["FINANCE"]
        self.assertEqual(fin_summary["loan_count"], 1)
        self.assertEqual(fin_summary["total_exposure"], 50000)
        self.assertAlmostEqual(fin_summary["total_expected_loss"], 24000.00)
        self.assertAlmostEqual(fin_summary["average_pd"], 0.6)
        self.assertAlmostEqual(fin_summary["average_lgd"], 0.8)
        self.assertEqual(fin_summary["defaulted_loan_count"],1)


    def test_get_risk_summary_by_country(self):
        """Test aggregation of risk by country."""
        portfolio_overview = [
            {"loan_id": "L1", "company_id": "C1", "industry_sector": "TECHNOLOGY", "country_iso_code": "US",
             "exposure_at_default_usd": 100000, "pd_estimate": 0.1, "lgd_estimate": 0.4,
             "expected_loss_usd": 4000.00, "is_defaulted": False},
            {"loan_id": "L2", "company_id": "C2", "industry_sector": "FINANCE", "country_iso_code": "GB",
             "exposure_at_default_usd": 50000, "pd_estimate": 0.6, "lgd_estimate": 0.8,
             "expected_loss_usd": 24000.00, "is_defaulted": True},
            {"loan_id": "L3", "company_id": "C3", "industry_sector": "TECHNOLOGY", "country_iso_code": "US",
             "exposure_at_default_usd": 200000, "pd_estimate": 0.05, "lgd_estimate": 0.5,
             "expected_loss_usd": 5000.00, "is_defaulted": False}
        ]
        summary = self.risk_map_service.get_risk_summary_by_country(portfolio_overview)

        self.assertIn("US", summary)
        self.assertIn("GB", summary)

        us_summary = summary["US"]
        self.assertEqual(us_summary["loan_count"], 2)
        self.assertEqual(us_summary["total_exposure"], 100000 + 200000)
        self.assertAlmostEqual(us_summary["total_expected_loss"], 4000.00 + 5000.00)
        self.assertAlmostEqual(us_summary["average_pd"], (0.1 + 0.05) / 2)
        self.assertAlmostEqual(us_summary["average_lgd"], (0.4 + 0.5) / 2)

        gb_summary = summary["GB"]
        self.assertEqual(gb_summary["loan_count"], 1)


if __name__ == '__main__':
    unittest.main()
