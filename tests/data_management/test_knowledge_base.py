import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import json
from pathlib import Path
import datetime
import io # Added for StringIO

from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import (
    CorporateEntity, LoanAgreement, FinancialStatement, DefaultEvent,
    IndustrySector, Currency, CollateralType # Removed SeniorityOfDebt and DefaultType
)

class TestKnowledgeBaseService(unittest.TestCase):

    # Sample data for tests
    sample_companies_csv_content = (
        "company_id,company_name,industry_sector,country_iso_code,founded_date,subsidiaries,suppliers,customers,loan_agreement_ids,financial_statement_ids,management_quality_score,market_share_percentage,esg_score\n"
        "C1,Corp One,Technology,US,2000-01-01,C2;C3,S1,CUST1;CUST2,L1,FS1,8,25,70\n" # Changed TECHNOLOGY to Technology
        "C2,Corp Two,Financial Services,GB,2005-06-15,,,,L2,FS2,7,10,60\n" # Changed FINANCE to Financial Services
        "C3,Corp Three,Technology,US,2010-11-01,,,,,,,\n" # Changed TECHNOLOGY to Technology
    )
    sample_loans_json_content = json.dumps([
        {
            "loan_id": "L1", "company_id": "C1", "loan_amount": 100000, "currency": "USD",
            "origination_date": "2022-01-01", "maturity_date": "2025-01-01",
            "interest_rate_percentage": 5.0, "collateral_type": "Real Estate", # Changed to enum value
            "seniority_of_debt": "SENIOR", "default_status": False, "guarantors": ["G1"],
            "collateral_value_usd": 120000, "recovery_rate_percentage": 0.6, "economic_condition_indicator": 0.5
        },
        {
            "loan_id": "L2", "company_id": "C2", "loan_amount": 50000, "currency": "GBP",
            "origination_date": "2023-01-01", "maturity_date": "2024-01-01",
            "interest_rate_percentage": 7.5, "collateral_type": "None", # Changed to enum value
            "seniority_of_debt": "JUNIOR", "default_status": True
        }
    ])
    sample_fs_json_content = json.dumps([
        {
            "statement_id": "FS1", "company_id": "C1", "statement_date": "2023-12-31",
            "currency": "USD", "revenue": 1000000, "net_income": 100000,
            "total_assets_usd": 750000, "total_liabilities_usd": 250000, "net_equity_usd": 500000, # Was 'total_assets'
            "reporting_period_months": 12, # Added
            "current_assets": 200000, "current_liabilities": 50000
        },
        {
            "statement_id": "FS2", "company_id": "C2", "statement_date": "2023-12-31",
            "currency": "GBP", "revenue": 500000, "net_income": 20000,
            "total_assets_usd": 300000, # Added
            "total_liabilities_usd": 150000, # Added
            "net_equity_usd": 150000, # Added
            "reporting_period_months": 12 # Added
        }
    ])
    sample_de_json_content = json.dumps([
        {
            "event_id": "DE1", "loan_id": "L2", "company_id": "C2",
            "default_date": "2023-07-01", "default_type": "MISSED_PAYMENT", # Changed to string
            "amount_at_default": 45000
        }
    ])

    @patch('pandas.read_csv') # as mock_pd_read_csv_in_setup
    @patch('builtins.open', new_callable=mock_open) # as mock_file_open_in_setup
    def setUp(self, mock_open_method_for_setup, mock_read_csv_method_for_setup): # Corrected arg names to match decorator order
        # Create the DataFrame that the mocked pd.read_csv will return
        self.mocked_companies_df = pd.read_csv(io.StringIO(self.sample_companies_csv_content))
        assert not self.mocked_companies_df.empty, "Mocked companies DataFrame is empty in setUp!" # Debug assertion
        mock_read_csv_method_for_setup.return_value = self.mocked_companies_df

        # This side_effect is for self.kb_service used by most tests
        def global_side_effect_open(file_path_str, mode='r'):
            file_path_name = Path(file_path_str).name
            if file_path_name == "sample_loans.json":
                # mock_open() returns the mock file handle
                return mock_open(read_data=self.sample_loans_json_content)()
            elif file_path_name == "sample_financial_statements.json":
                return mock_open(read_data=self.sample_fs_json_content)()
            elif file_path_name == "sample_default_events.json":
                return mock_open(read_data=self.sample_de_json_content)()
            # pd.read_csv is mocked separately, so open shouldn't be called for sample_companies.csv by KnowledgeBaseService
            # However, if a test *directly* tried to open it via builtins.open, this could handle it:
            elif file_path_name == "sample_companies.csv":
                 # This case should ideally not be reached if pd.read_csv is properly mocked for company data
                 return mock_open(read_data=self.sample_companies_csv_content)()
            # Fallback for any other unexpected file open attempts through the mock
            # print(f"Warning: Unexpected file opened in global setUp mock: {file_path_str}")
            mf = mock_open(read_data="{}") # Default empty JSON for other unexpected calls
            return mf()


        # unittest.mock.mock_open is the callable, the argument mock_open_method_for_setup is the instance of the mock for 'open'
        # The side_effect should be set on the instance.
        mock_open_method_for_setup.side_effect = global_side_effect_open

        # Initialize KnowledgeBaseService - this will trigger _load_data with mocked file reads
        # The paths here should match what global_side_effect_open expects by name
        self.kb_service = KnowledgeBaseService(
            companies_data_path=Path("data/sample_companies.csv"),
            loans_data_path=Path("data/sample_loans.json"),
            financial_statements_path=Path("data/sample_financial_statements.json"),
            default_events_path=Path("data/sample_default_events.json")
        )

    def test_load_data_companies(self):
        """Test that company data is loaded and parsed correctly."""
        self.assertIsNotNone(self.kb_service._companies_df)
        self.assertEqual(len(self.kb_service._companies_df), 3)

        # Check C1 (Corp One) from DataFrame
        c1_data = self.kb_service._companies_df[self.kb_service._companies_df['company_id'] == 'C1'].iloc[0]
        self.assertEqual(c1_data['company_name'], 'Corp One')
        self.assertEqual(c1_data['industry_sector'], IndustrySector.TECHNOLOGY.value)
        self.assertEqual(c1_data['founded_date'], datetime.date(2000, 1, 1))
        self.assertListEqual(c1_data['subsidiaries'], ['C2', 'C3'])
        self.assertListEqual(c1_data['suppliers'], ['S1'])
        self.assertListEqual(c1_data['customers'], ['CUST1', 'CUST2'])
        self.assertEqual(c1_data['management_quality_score'], 8)

        # Check C3 (Corp Three) for handling of missing optional values (should be None or empty list)
        c3_data = self.kb_service._companies_df[self.kb_service._companies_df['company_id'] == 'C3'].iloc[0]
        self.assertIsNone(c3_data['subsidiaries']) # Based on current _load_data logic for empty strings
        self.assertIsNone(c3_data['management_quality_score'])


    def test_load_data_loans(self):
        """Test that loan data is loaded and parsed correctly."""
        self.assertIsNotNone(self.kb_service._loans_data)
        self.assertEqual(len(self.kb_service._loans_data), 2)
        l1_data = next(loan for loan in self.kb_service._loans_data if loan.loan_id == 'L1')
        self.assertIsInstance(l1_data, LoanAgreement)
        self.assertEqual(l1_data.company_id, 'C1')
        self.assertEqual(l1_data.loan_amount, 100000)
        self.assertEqual(l1_data.origination_date, datetime.date(2022, 1, 1))
        self.assertEqual(l1_data.collateral_type, CollateralType.REAL_ESTATE)
        self.assertFalse(l1_data.default_status)
        self.assertListEqual(l1_data.guarantors, ["G1"])

    def test_load_data_financial_statements(self):
        """Test that financial statement data is loaded correctly."""
        self.assertIsNotNone(self.kb_service._financial_statements_data)
        self.assertEqual(len(self.kb_service._financial_statements_data), 2)
        fs1_data = next(fs for fs in self.kb_service._financial_statements_data if fs.statement_id == 'FS1')
        self.assertIsInstance(fs1_data, FinancialStatement)
        self.assertEqual(fs1_data.company_id, 'C1')
        self.assertEqual(fs1_data.revenue, 1000000)
        self.assertEqual(fs1_data.statement_date, datetime.date(2023,12,31))

    def test_load_data_default_events(self):
        """Test that default event data is loaded correctly."""
        self.assertIsNotNone(self.kb_service._default_events_data)
        self.assertEqual(len(self.kb_service._default_events_data), 1)
        de1_data = self.kb_service._default_events_data[0]
        self.assertIsInstance(de1_data, DefaultEvent)
        self.assertEqual(de1_data.loan_id, 'L2')
        self.assertEqual(de1_data.default_type, "MISSED_PAYMENT") # Ensure this is a string
        self.assertEqual(de1_data.default_date, datetime.date(2023,7,1))


    def test_get_all_companies(self):
        """Test retrieving all companies."""
        all_companies = self.kb_service.get_all_companies()
        self.assertEqual(len(all_companies), 3)
        self.assertTrue(all(isinstance(c, CorporateEntity) for c in all_companies))

    def test_get_all_companies_filtered(self):
        """Test retrieving companies with filters."""
        tech_companies = self.kb_service.get_all_companies(industry_sector=IndustrySector.TECHNOLOGY)
        self.assertEqual(len(tech_companies), 2) # C1 and C3
        self.assertTrue(all(c.industry_sector == IndustrySector.TECHNOLOGY for c in tech_companies))

        us_companies = self.kb_service.get_all_companies(country_iso_code="US")
        self.assertEqual(len(us_companies), 2) # C1 and C3

        gb_finance_companies = self.kb_service.get_all_companies(industry_sector=IndustrySector.FINANCIAL_SERVICES, country_iso_code="GB") # Changed to FINANCIAL_SERVICES
        self.assertEqual(len(gb_finance_companies), 1) # C2
        self.assertEqual(gb_finance_companies[0].company_id, "C2")


    def test_get_company_profile(self):
        """Test retrieving a single company profile."""
        c1_profile = self.kb_service.get_company_profile("C1")
        self.assertIsNotNone(c1_profile)
        self.assertIsInstance(c1_profile, CorporateEntity)
        self.assertEqual(c1_profile.company_name, "Corp One")

        non_existent = self.kb_service.get_company_profile("NON_EXISTENT")
        self.assertIsNone(non_existent)

    def test_get_all_loans(self):
        """Test retrieving all loans."""
        all_loans = self.kb_service.get_all_loans()
        self.assertEqual(len(all_loans), 2)
        self.assertTrue(all(isinstance(l, LoanAgreement) for l in all_loans))

    def test_get_loans_for_company(self):
        """Test retrieving loans for a specific company."""
        c1_loans = self.kb_service.get_loans_for_company("C1")
        self.assertEqual(len(c1_loans), 1)
        self.assertEqual(c1_loans[0].loan_id, "L1")

        c3_loans = self.kb_service.get_loans_for_company("C3") # C3 has no loans in sample
        self.assertEqual(len(c3_loans), 0)

    def test_get_financial_statements_for_company(self):
        """Test retrieving financial statements for a company."""
        c1_fs = self.kb_service.get_financial_statements_for_company("C1")
        self.assertEqual(len(c1_fs), 1)
        self.assertEqual(c1_fs[0].statement_id, "FS1")

        c3_fs = self.kb_service.get_financial_statements_for_company("C3") # C3 has no FS
        self.assertEqual(len(c3_fs), 0)

    def test_get_default_events_for_loan(self):
        """Test retrieving default events for a loan."""
        l2_de = self.kb_service.get_default_events_for_loan("L2")
        self.assertEqual(len(l2_de), 1)
        self.assertEqual(l2_de[0].event_id, "DE1")

        l1_de = self.kb_service.get_default_events_for_loan("L1") # L1 has no default events
        self.assertEqual(len(l1_de), 0)

    def test_get_loans_by_criteria(self):
        """Test retrieving loans by various criteria."""
        # Default status True
        defaulted_loans = self.kb_service.get_loans_by_criteria(default_status=True)
        self.assertEqual(len(defaulted_loans), 1)
        self.assertEqual(defaulted_loans[0].loan_id, "L2")

        # Currency GBP
        gbp_loans = self.kb_service.get_loans_by_criteria(currency=Currency.GBP) # Changed to Currency
        self.assertEqual(len(gbp_loans), 1)
        self.assertEqual(gbp_loans[0].loan_id, "L2")

        # Min amount
        min_amount_loans = self.kb_service.get_loans_by_criteria(min_amount=60000)
        self.assertEqual(len(min_amount_loans), 1)
        self.assertEqual(min_amount_loans[0].loan_id, "L1")

        # Max amount
        max_amount_loans = self.kb_service.get_loans_by_criteria(max_amount=60000)
        self.assertEqual(len(max_amount_loans), 1)
        self.assertEqual(max_amount_loans[0].loan_id, "L2")

        # Collateral Type
        real_estate_loans = self.kb_service.get_loans_by_criteria(collateral_type=CollateralType.REAL_ESTATE)
        self.assertEqual(len(real_estate_loans), 1)
        self.assertEqual(real_estate_loans[0].loan_id, "L1")

        # Combined criteria
        combined_loans = self.kb_service.get_loans_by_criteria(currency=Currency.USD, default_status=False) # Changed to Currency
        self.assertEqual(len(combined_loans), 1)
        self.assertEqual(combined_loans[0].loan_id, "L1")


    @patch('pandas.read_csv', side_effect=FileNotFoundError("Mocked File Not Found"))
    def test_load_data_company_file_not_found(self, mock_read_csv):
        """Test handling when company data file is not found."""
        # Must re-initialize to trigger _load_data with new mock
        with patch('builtins.open', mock_open(read_data=self.sample_loans_json_content)): # Mock other files
            kb_service_test = KnowledgeBaseService(companies_data_path=Path("non_existent.csv"))
        self.assertIsNotNone(kb_service_test._companies_df)
        self.assertTrue(kb_service_test._companies_df.empty)
        self.assertEqual(len(kb_service_test.get_all_companies()), 0)
        # Other data should still load if their files are present (mocked here)
        self.assertEqual(len(kb_service_test.get_all_loans()), 2)


    def test_load_data_json_file_not_found(self):
        """Test handling when a JSON data file is not found (e.g., loans)."""

        # Mock pandas.read_csv to return the sample company data successfully
        mock_csv_df = pd.read_csv(io.StringIO(self.sample_companies_csv_content))

        def open_side_effect(file_path_str, mode='r'):
            file_path = Path(file_path_str)
            if file_path.name == "non_existent_loans.json":
                raise FileNotFoundError("Mocked: non_existent_loans.json not found")
            elif file_path.name == "sample_financial_statements.json":
                return mock_open(read_data=self.sample_fs_json_content)()
            elif file_path.name == "sample_default_events.json":
                return mock_open(read_data=self.sample_de_json_content)()
            # This case should ideally not be hit if companies_data_path is also mocked or pd.read_csv is mocked
            elif file_path.name == "sample_companies.csv": # Should be handled by pd.read_csv mock
                 return mock_open(read_data=self.sample_companies_csv_content)()
            raise ValueError(f"Unexpected file open in this test: {file_path_str}")

        with patch('pandas.read_csv', return_value=mock_csv_df) as mock_pd_read_csv_specific, \
             patch('builtins.open', side_effect=open_side_effect) as mock_open_specific:

            kb_service_test = KnowledgeBaseService(
                companies_data_path=Path("data/sample_companies.csv"), # Path used by read_csv mock
                loans_data_path=Path("data/non_existent_loans.json"), # This will trigger FileNotFoundError
                financial_statements_path=Path("data/sample_financial_statements.json"),
                default_events_path=Path("data/sample_default_events.json")
            )

        self.assertIsNotNone(kb_service_test._loans_data) # Should be initialized to []
        self.assertEqual(len(kb_service_test._loans_data), 0)
        self.assertEqual(len(kb_service_test.get_all_loans()), 0)

        # Companies, FS, and DE should still load if their files are "found" by the mock
        self.assertEqual(len(kb_service_test.get_all_companies()), 3)
        self.assertTrue(len(kb_service_test._financial_statements_data) > 0)
        self.assertTrue(len(kb_service_test._default_events_data) > 0)


if __name__ == '__main__':
    unittest.main()
