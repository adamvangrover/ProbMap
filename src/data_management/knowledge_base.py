import pandas as pd
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.core.config import settings # Using settings for data paths
from src.data_management.ontology import CorporateEntity, LoanAgreement # Import Pydantic models

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """
    Manages access to the foundational data for credit analysis.
    For PoC, this service reads from local CSV/JSON files.
    In a production system, this would interact with a database like Vega.
    """
    def __init__(self, companies_data_path: Path = None, loans_data_path: Path = None):
        base_data_dir = Path("data") # Default base directory
        self.companies_data_path = companies_data_path or base_data_dir / "sample_companies.csv"
        self.loans_data_path = loans_data_path or base_data_dir / "sample_loans.json"

        self._companies_df: Optional[pd.DataFrame] = None
        self._loans_data: Optional[List[Dict[str, Any]]] = None

        self._load_data()

    def _load_data(self):
        """Loads data from the configured paths."""
        try:
            logger.info(f"Loading company data from: {self.companies_data_path}")
            self._companies_df = pd.read_csv(self.companies_data_path)
            # Convert relevant columns to appropriate types if necessary
            self._companies_df['founded_date'] = pd.to_datetime(self._companies_df['founded_date'], errors='coerce').dt.date
            logger.info(f"Loaded {len(self._companies_df)} company records.")
        except FileNotFoundError:
            logger.error(f"Company data file not found at {self.companies_data_path}. No company data loaded.")
            self._companies_df = pd.DataFrame() # Empty DataFrame
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            self._companies_df = pd.DataFrame()

        try:
            logger.info(f"Loading loan data from: {self.loans_data_path}")
            with open(self.loans_data_path, 'r') as f:
                self._loans_data = json.load(f)
            # Optional: Validate loans data with Pydantic models
            # validated_loans = [LoanAgreement(**loan) for loan in self._loans_data]
            # self._loans_data = [loan.model_dump() for loan in validated_loans]
            logger.info(f"Loaded {len(self._loans_data)} loan records.")
        except FileNotFoundError:
            logger.error(f"Loan data file not found at {self.loans_data_path}. No loan data loaded.")
            self._loans_data = []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.loans_data_path}: {e}")
            self._loans_data = []
        except Exception as e: # Catch other exceptions like Pydantic validation errors
            logger.error(f"Error loading or validating loan data: {e}")
            self._loans_data = []

    def get_company_profile(self, company_id: str) -> Optional[CorporateEntity]:
        """Retrieves a company profile by its ID."""
        if self._companies_df is None or self._companies_df.empty:
            logger.warning("Company data is not loaded.")
            return None

        company_series = self._companies_df[self._companies_df['company_id'] == company_id].iloc[0]
        if not company_series.empty:
            # Pandas to_dict might introduce NaT which Pydantic needs to handle or be None
            company_dict = company_series.where(pd.notnull(company_series), None).to_dict()
            try:
                return CorporateEntity(**company_dict)
            except Exception as e:
                logger.error(f"Error creating CorporateEntity for {company_id}: {e}. Data: {company_dict}")
                return None
        logger.warning(f"Company with ID {company_id} not found.")
        return None

    def get_all_companies(self) -> List[CorporateEntity]:
        """Retrieves all company profiles."""
        if self._companies_df is None or self._companies_df.empty:
            return []

        companies = []
        for _, row in self._companies_df.iterrows():
            company_dict = row.where(pd.notnull(row), None).to_dict()
            try:
                companies.append(CorporateEntity(**company_dict))
            except Exception as e:
                logger.error(f"Error creating CorporateEntity for row: {e}. Data: {company_dict}")
        return companies

    def get_loans_for_company(self, company_id: str) -> List[LoanAgreement]:
        """Retrieves all loans associated with a specific company ID."""
        if not self._loans_data:
            return []

        company_loans = []
        for loan_dict in self._loans_data:
            if loan_dict.get('company_id') == company_id:
                try:
                    company_loans.append(LoanAgreement(**loan_dict))
                except Exception as e:
                    logger.error(f"Error creating LoanAgreement for loan {loan_dict.get('loan_id')}: {e}. Data: {loan_dict}")
        return company_loans

    def get_all_loans(self) -> List[LoanAgreement]:
        """Retrieves all loan records."""
        if not self._loans_data:
            return []

        all_loans = []
        for loan_dict in self._loans_data:
            try:
                all_loans.append(LoanAgreement(**loan_dict))
            except Exception as e:
                logger.error(f"Error creating LoanAgreement for loan {loan_dict.get('loan_id')}: {e}. Data: {loan_dict}")
        return all_loans

    # Conceptual methods for a production system with Vega DB
    def store_company_profile(self, company: CorporateEntity) -> bool:
        logger.info(f"PoC: Simulating storing company: {company.company_id}")
        # In production: self.vega_client.insert_company(company.model_dump())
        # For PoC, could append to CSV or update DataFrame, but requires careful handling of file I/O.
        # For simplicity, this PoC focuses on reads.
        return True

    def store_loan_agreement(self, loan: LoanAgreement) -> bool:
        logger.info(f"PoC: Simulating storing loan: {loan.loan_id}")
        # In production: self.vega_client.insert_loan(loan.model_dump())
        return True

if __name__ == "__main__":
    # This requires src.core.logging_config to be importable,
    # which means running from the project root with `python -m src.data_management.knowledge_base`
    # or ensuring PYTHONPATH is set up correctly.

    # Assuming running from project root:
    # from src.core.logging_config import setup_logging # Explicitly call if not auto-setup
    # setup_logging() # If not already called on import in logging_config

    kb_service = KnowledgeBaseService()

    logger.info("--- Testing KnowledgeBaseService ---")

    # Test get_company_profile
    test_company_id = "COMP001"
    company = kb_service.get_company_profile(test_company_id)
    if company:
        logger.info(f"Found company {test_company_id}: {company.company_name}")
    else:
        logger.warning(f"Company {test_company_id} not found.")

    non_existent_company_id = "COMP999"
    company_none = kb_service.get_company_profile(non_existent_company_id)
    if not company_none:
        logger.info(f"Correctly did not find non-existent company {non_existent_company_id}.")

    # Test get_all_companies
    all_companies = kb_service.get_all_companies()
    logger.info(f"Total companies loaded: {len(all_companies)}")
    if all_companies:
        logger.info(f"First loaded company: {all_companies[0].company_name}")

    # Test get_loans_for_company
    loans_for_comp1 = kb_service.get_loans_for_company("COMP001")
    logger.info(f"Loans for COMP001: {len(loans_for_comp1)}")
    if loans_for_comp1:
        logger.info(f"First loan for COMP001 amount: {loans_for_comp1[0].loan_amount}")

    # Test get_all_loans
    all_loans = kb_service.get_all_loans()
    logger.info(f"Total loans loaded: {len(all_loans)}")
    if all_loans:
        logger.info(f"First loaded loan ID: {all_loans[0].loan_id}, Default Status: {all_loans[0].default_status}")

    # Test with non-existent files (manual test by renaming files)
    # logger.info("\n--- Testing with non-existent files (manual test needed) ---")
    # kb_service_missing = KnowledgeBaseService(
    #     companies_data_path=Path("data/missing_companies.csv"),
    #     loans_data_path=Path("data/missing_loans.json")
    # )
    # logger.info(f"Companies loaded with missing file: {len(kb_service_missing.get_all_companies())}")
    # logger.info(f"Loans loaded with missing file: {len(kb_service_missing.get_all_loans())}")
