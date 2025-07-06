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
    LoanAgreement, CorporateEntity, IndustrySector, Currency, CollateralType, RiskItem, HITLAnnotation # Added RiskItem, HITLAnnotation
)

class TestRiskMapService(unittest.TestCase):

    def setUp(self):
        self.mock_kb_service = MagicMock(spec=KnowledgeBaseService)
        self.mock_pd_model = MagicMock(spec=PDModel)
        self.mock_lgd_model = MagicMock(spec=LGDModel)
        self.mock_kg_service = MagicMock(spec=KnowledgeGraphService)
        self.mock_kg_service.graph = MagicMock(spec=nx.MultiDiGraph)

        type(self.mock_pd_model).model = PropertyMock(return_value=True)
        type(self.mock_lgd_model).model = PropertyMock(return_value=True)

        # Default mock for get_hitl_annotation to return None
        self.mock_kb_service.get_hitl_annotation.return_value = None

        self.risk_map_service = RiskMapService(
            kb_service=self.mock_kb_service,
            pd_model=self.mock_pd_model,
            lgd_model=self.mock_lgd_model,
            kg_service=self.mock_kg_service
        )

        self.company1 = CorporateEntity(
            company_id="C1", company_name="Corp Alpha", industry_sector=IndustrySector.TECHNOLOGY,
            country_iso_code="US", founded_date=datetime.date(2000,1,1), management_quality_score=8
        )
        self.loan1 = LoanAgreement(
            loan_id="L1", company_id="C1", loan_amount=100000, currency=Currency.USD,
            origination_date=datetime.date(2022,1,1), maturity_date=datetime.date(2025,1,1),
            interest_rate_percentage=5.0, collateral_type=CollateralType.REAL_ESTATE,
            seniority_of_debt="Senior", default_status=False,
            economic_condition_indicator=0.6
        )
        self.company2 = CorporateEntity(
            company_id="C2", company_name="Corp Beta", industry_sector=IndustrySector.FINANCIAL_SERVICES,
            country_iso_code="GB", founded_date=datetime.date(2005,1,1), management_quality_score=6
        )
        self.loan2 = LoanAgreement(
            loan_id="L2", company_id="C2", loan_amount=50000, currency=Currency.GBP,
            origination_date=datetime.date(2023,1,1), maturity_date=datetime.date(2024,1,1),
            interest_rate_percentage=7.0, collateral_type=CollateralType.NONE,
            seniority_of_debt="Junior", default_status=True,
            economic_condition_indicator=0.3
        )
        self.company3 = CorporateEntity( # For filtering tests
            company_id="C3", company_name="Corp Gamma", industry_sector=IndustrySector.TECHNOLOGY,
            country_iso_code="DE", founded_date=datetime.date(2010,1,1), management_quality_score=7
        )
        self.loan3 = LoanAgreement(
            loan_id="L3", company_id="C3", loan_amount=200000, currency=Currency.EUR,
            origination_date=datetime.date(2021,1,1), maturity_date=datetime.date(2026,1,1),
            interest_rate_percentage=3.0, collateral_type=CollateralType.EQUIPMENT,
            seniority_of_debt="Senior", default_status=False,
            economic_condition_indicator=0.7
        )

        # Sample HITL Annotation
        self.hitl_c1_company = HITLAnnotation(
            entity_id="C1", annotation_type="company", hitl_management_quality_score=9,
            hitl_review_status="Reviewed - Improved", annotator_id="test_user"
        )
        self.hitl_l2_loan = HITLAnnotation(
            entity_id="L2", annotation_type="loan", hitl_pd_override=0.65,
            hitl_review_status="Flagged - PD Override", hitl_analyst_notes="Analyst adjusted PD based on new info."
        )


    def test_generate_portfolio_risk_overview_successful(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2]
        def get_company_profile_side_effect(company_id):
            if company_id == "C1": return self.company1
            if company_id == "C2": return self.company2
            return None
        self.mock_kb_service.get_company_profile.side_effect = get_company_profile_side_effect

        # Mock HITL annotations
        def get_hitl_annotation_side_effect(entity_id, annotation_type):
            if entity_id == "C1" and annotation_type == "company": return self.hitl_c1_company
            if entity_id == "L2" and annotation_type == "loan": return self.hitl_l2_loan
            return None
        self.mock_kb_service.get_hitl_annotation.side_effect = get_hitl_annotation_side_effect

        self.mock_pd_model.predict_for_loan.side_effect = [(0, 0.1), (1, 0.6)]
        self.mock_lgd_model.predict_lgd.side_effect = [0.4, 0.8]
        self.mock_kg_service.graph.has_node.return_value = True
        self.mock_kg_service.get_company_contextual_info.side_effect = [
            {"degree_centrality": 0.5, "num_suppliers": 5, "num_customers": 10, "num_subsidiaries": 2},
            {"degree_centrality": 0.3, "num_suppliers": 2, "num_customers": 3, "num_subsidiaries": 0}
        ]

        overview = self.risk_map_service.generate_portfolio_risk_overview()
        self.assertEqual(len(overview), 2)
        self.assertTrue(all(isinstance(item, RiskItem) for item in overview))

        item1 = next(item for item in overview if item.loan_id == "L1")
        self.assertEqual(item1.company_id, "C1")
        self.assertEqual(item1.pd_estimate, 0.1)
        self.assertEqual(item1.lgd_estimate, 0.4)
        self.assertEqual(item1.exposure_at_default_usd, self.loan1.loan_amount)
        expected_el1 = 0.1 * 0.4 * self.loan1.loan_amount
        self.assertAlmostEqual(item1.expected_loss_usd, round(expected_el1, 2))
        self.assertEqual(item1.industry_sector, IndustrySector.TECHNOLOGY.value)
        self.assertEqual(item1.country_iso_code, "US")
        self.assertEqual(item1.kg_degree_centrality, 0.5)
        self.assertEqual(item1.kg_num_suppliers, 5)
        self.assertFalse(item1.is_defaulted)
        self.assertEqual(item1.management_quality_score, 8) # Original MQS
        self.assertEqual(item1.hitl_management_quality_score, 9) # Overridden MQS
        self.assertEqual(item1.hitl_review_status, "Reviewed - Improved") # From company annotation for L1/C1
        self.assertIsNone(item1.hitl_pd_override)


        item2 = next(item for item in overview if item.loan_id == "L2")
        self.assertEqual(item2.company_id, "C2")
        self.assertEqual(item2.pd_estimate, 0.6)
        self.assertEqual(item2.lgd_estimate, 0.8)
        expected_el2 = 0.6 * 0.8 * self.loan2.loan_amount
        self.assertAlmostEqual(item2.expected_loss_usd, round(expected_el2, 2))
        self.assertTrue(item2.is_defaulted)
        self.assertEqual(item2.hitl_pd_override, 0.65) # From loan annotation
        self.assertEqual(item2.hitl_review_status, "Flagged - PD Override") # From loan annotation
        self.assertEqual(item2.management_quality_score, 6) # Original MQS
        self.assertEqual(item2.hitl_management_quality_score, 6) # No company override for C2, so defaults to original


    def test_generate_overview_pd_prediction_none(self): # Renamed for clarity
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = None
        self.mock_lgd_model.predict_lgd.return_value = 0.4
        self.mock_kg_service.graph.has_node.return_value = False
        self.mock_kb_service.get_hitl_annotation.return_value = None


        overview = self.risk_map_service.generate_portfolio_risk_overview()
        self.assertEqual(len(overview), 1)
        item1 = overview[0]
        self.assertIsNone(item1.pd_estimate)
        self.assertIsNone(item1.expected_loss_usd) # EL should be None if PD is None


    def test_generate_overview_lgd_model_predict_none(self): # Renamed for clarity
        type(self.mock_lgd_model).model = PropertyMock(return_value=True) # Ensure model is "loaded"
        self.mock_lgd_model.predict_lgd.return_value = None # Simulate predict_lgd returning None (though current type hint is float)

        risk_map_service_test_lgd = RiskMapService(
            self.mock_kb_service, self.mock_pd_model, self.mock_lgd_model, self.mock_kg_service
        )
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1)
        self.mock_kg_service.graph.has_node.return_value = False
        self.mock_kb_service.get_hitl_annotation.return_value = None


        overview = risk_map_service_test_lgd.generate_portfolio_risk_overview()
        self.assertEqual(len(overview), 1)
        item1 = overview[0]
        self.assertIsNone(item1.lgd_estimate) # LGD is None
        self.assertIsNone(item1.expected_loss_usd) # EL should be None if LGD is None


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
        self.assertIsNone(item1.kg_degree_centrality) # KG service mock returns has_node=False
        self.assertIsNone(item1.kg_num_suppliers)


    def test_get_risk_summary_by_sector(self):
        # Mock generate_portfolio_risk_overview to return a list of RiskItem objects
        # This test now focuses on the summary logic, assuming generate_portfolio_risk_overview works
        # (tested separately for its RiskItem generation)

        # Create RiskItem instances for testing summary
        item1 = RiskItem(loan_id="L1", company_id="C1", company_name="Alpha", industry_sector=IndustrySector.TECHNOLOGY.value,
                         country_iso_code="US", loan_amount_usd=1000, currency="USD", is_defaulted=False,
                         pd_estimate=0.1, lgd_estimate=0.4, exposure_at_default_usd=1000, expected_loss_usd=40)
        item2 = RiskItem(loan_id="L2", company_id="C2", company_name="Beta", industry_sector=IndustrySector.FINANCIAL_SERVICES.value,
                         country_iso_code="GB", loan_amount_usd=500, currency="GBP", is_defaulted=True,
                         pd_estimate=0.6, lgd_estimate=0.8, exposure_at_default_usd=500, expected_loss_usd=240)
        item3 = RiskItem(loan_id="L3", company_id="C3", company_name="Gamma", industry_sector=IndustrySector.TECHNOLOGY.value,
                         country_iso_code="US", loan_amount_usd=2000, currency="USD", is_defaulted=False,
                         pd_estimate=0.05, lgd_estimate=0.5, exposure_at_default_usd=2000, expected_loss_usd=50)

        mock_portfolio_overview = [item1, item2, item3]

        summary = self.risk_map_service.get_risk_summary_by_sector(mock_portfolio_overview)

        self.assertIn(IndustrySector.TECHNOLOGY.value, summary)
        self.assertIn(IndustrySector.FINANCIAL_SERVICES.value, summary)

        tech_summary = summary[IndustrySector.TECHNOLOGY.value]
        self.assertEqual(tech_summary["loan_count"], 2)
        self.assertEqual(tech_summary["total_exposure"], 1000 + 2000)
        self.assertAlmostEqual(tech_summary["total_expected_loss"], 40.00 + 50.00)
        self.assertAlmostEqual(tech_summary["average_pd"], (0.1 + 0.05) / 2)
        self.assertAlmostEqual(tech_summary["average_lgd"], (0.4 + 0.5) / 2)
        self.assertEqual(tech_summary["defaulted_loan_count"],0)

        fin_summary = summary[IndustrySector.FINANCIAL_SERVICES.value]
        self.assertEqual(fin_summary["loan_count"], 1)
        self.assertEqual(fin_summary["total_exposure"], 500)
        self.assertAlmostEqual(fin_summary["total_expected_loss"], 240.00)
        self.assertAlmostEqual(fin_summary["average_pd"], 0.6)
        self.assertAlmostEqual(fin_summary["average_lgd"], 0.8)
        self.assertEqual(fin_summary["defaulted_loan_count"],1)

    def test_get_risk_summary_by_country(self):
        item1 = RiskItem(loan_id="L1", company_id="C1", company_name="Alpha", industry_sector="Tech", country_iso_code="US",
                         loan_amount_usd=1000, currency="USD", is_defaulted=False, pd_estimate=0.1, lgd_estimate=0.4,
                         exposure_at_default_usd=1000, expected_loss_usd=40)
        item2 = RiskItem(loan_id="L2", company_id="C2", company_name="Beta", industry_sector="Finance", country_iso_code="GB",
                         loan_amount_usd=500, currency="GBP", is_defaulted=True, pd_estimate=0.6, lgd_estimate=0.8,
                         exposure_at_default_usd=500, expected_loss_usd=240)
        item3 = RiskItem(loan_id="L3", company_id="C3", company_name="Gamma", industry_sector="Tech", country_iso_code="US",
                         loan_amount_usd=2000, currency="USD", is_defaulted=False, pd_estimate=0.05, lgd_estimate=0.5,
                         exposure_at_default_usd=2000, expected_loss_usd=50)
        mock_portfolio_overview = [item1, item2, item3]
        summary = self.risk_map_service.get_risk_summary_by_country(mock_portfolio_overview)

        self.assertIn("US", summary)
        self.assertIn("GB", summary)
        us_summary = summary["US"]
        self.assertEqual(us_summary["loan_count"], 2)
        gb_summary = summary["GB"]
        self.assertEqual(gb_summary["loan_count"], 1)

    def test_filtering_by_industry_sector(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3]
        def get_company_profile_side_effect(company_id):
            if company_id == "C1": return self.company1 # Tech
            if company_id == "C2": return self.company2 # Finance
            if company_id == "C3": return self.company3 # Tech
            return None
        self.mock_kb_service.get_company_profile.side_effect = get_company_profile_side_effect
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1) # Dummy PD
        self.mock_lgd_model.predict_lgd.return_value = 0.4 # Dummy LGD

        overview = self.risk_map_service.generate_portfolio_risk_overview(industry_sector=IndustrySector.TECHNOLOGY.value)
        self.assertEqual(len(overview), 2)
        self.assertTrue(all(item.industry_sector == IndustrySector.TECHNOLOGY.value for item in overview))
        loan_ids = {item.loan_id for item in overview}
        self.assertIn("L1", loan_ids)
        self.assertIn("L3", loan_ids)

    def test_filtering_by_country_iso_code(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3]
        def get_company_profile_side_effect(company_id):
            if company_id == "C1": return self.company1 # US
            if company_id == "C2": return self.company2 # GB
            if company_id == "C3": return self.company3 # DE
            return None
        self.mock_kb_service.get_company_profile.side_effect = get_company_profile_side_effect
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1)
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = self.risk_map_service.generate_portfolio_risk_overview(country_iso_code="US")
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0].loan_id, "L1")

    def test_filtering_by_loan_amount(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3] # Amounts: 100k, 50k, 200k
        self.mock_kb_service.get_company_profile.side_effect = [self.company1, self.company2, self.company3]
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.1)
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = self.risk_map_service.generate_portfolio_risk_overview(min_loan_amount_usd=60000, max_loan_amount_usd=150000)
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0].loan_id, "L1") # L1 is 100k

    def test_filtering_by_pd_estimate(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3]
        self.mock_kb_service.get_company_profile.side_effect = [self.company1, self.company2, self.company3]
        self.mock_pd_model.predict_for_loan.side_effect = [(0,0.05), (0,0.15), (0,0.25)] # L1=0.05, L2=0.15, L3=0.25
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = self.risk_map_service.generate_portfolio_risk_overview(min_pd_estimate=0.1, max_pd_estimate=0.2)
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0].loan_id, "L2") # L2 PD is 0.15

    def test_filtering_by_default_status(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2, self.loan3] # L2 is defaulted
        self.mock_kb_service.get_company_profile.side_effect = [self.company1, self.company2, self.company3]
        self.mock_pd_model.predict_for_loan.return_value = (0,0.1)
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = self.risk_map_service.generate_portfolio_risk_overview(default_status=True)
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0].loan_id, "L2")

    def test_filtering_by_hitl_review_status(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1, self.loan2]
        self.mock_kb_service.get_company_profile.side_effect = [self.company1, self.company2]
        self.mock_pd_model.predict_for_loan.return_value = (0,0.1)
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        # Mock HITL annotations
        def get_hitl_annotation_side_effect(entity_id, annotation_type):
            if entity_id == "L1" and annotation_type == "loan":
                return HITLAnnotation(entity_id="L1", annotation_type="loan", hitl_review_status="Reviewed")
            if entity_id == "L2" and annotation_type == "loan":
                return HITLAnnotation(entity_id="L2", annotation_type="loan", hitl_review_status="Flagged")
            return None
        self.mock_kb_service.get_hitl_annotation.side_effect = get_hitl_annotation_side_effect

        overview = self.risk_map_service.generate_portfolio_risk_overview(hitl_review_status="Flagged")
        self.assertEqual(len(overview), 1)
        self.assertEqual(overview[0].loan_id, "L2")
        self.assertEqual(overview[0].hitl_review_status, "Flagged")

    def test_filtering_no_results(self):
        self.mock_kb_service.get_all_loans.return_value = [self.loan1]
        self.mock_kb_service.get_company_profile.return_value = self.company1
        self.mock_pd_model.predict_for_loan.return_value = (0, 0.01) # Very low PD
        self.mock_lgd_model.predict_lgd.return_value = 0.4

        overview = self.risk_map_service.generate_portfolio_risk_overview(min_pd_estimate=0.9) # Filter for very high PD
        self.assertEqual(len(overview), 0)

if __name__ == '__main__':
    unittest.main()
