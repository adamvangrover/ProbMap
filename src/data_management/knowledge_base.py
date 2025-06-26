import pandas as pd
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from src.core.config import settings # Using settings for data paths
from src.data_management.ontology import CorporateEntity, LoanAgreement, FinancialStatement, DefaultEvent, IndustrySector, Currency, CollateralType # Import Pydantic models

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """
    Manages access to the foundational data for credit analysis.
    For PoC, this service reads from local CSV/JSON files.
    In a production system, this would interact with a database like Vega.
    """
    def __init__(self,
                 companies_data_path: Path = None,
                 loans_data_path: Path = None,
                 financial_statements_path: Path = None,
                 default_events_path: Path = None):
        base_data_dir = Path("data") # Default base directory
        self.companies_data_path = companies_data_path or base_data_dir / "sample_companies.csv"
        self.loans_data_path = loans_data_path or base_data_dir / "sample_loans.json"
        self.financial_statements_path = financial_statements_path or base_data_dir / "sample_financial_statements.json"
        self.default_events_path = default_events_path or base_data_dir / "sample_default_events.json"

        self._companies_df: Optional[pd.DataFrame] = None
        # Store validated Pydantic objects instead of raw dicts
        self._loans_data: Optional[List[LoanAgreement]] = None
        self._financial_statements_data: Optional[List[FinancialStatement]] = None
        self._default_events_data: Optional[List[DefaultEvent]] = None

        self._load_data()

    def _load_data(self):
        """Loads data from the configured paths."""
        # Load Companies
        try:
            logger.info(f"Loading company data from: {self.companies_data_path}")
            raw_companies_df = pd.read_csv(self.companies_data_path)

            if raw_companies_df.empty:
                logger.error(f"Company DataFrame loaded from {self.companies_data_path} is EMPTY.")

            validated_companies = []
            for _, row in raw_companies_df.iterrows():
                company_dict_raw = row.where(pd.notnull(row), None).to_dict()
                # Filter to include only fields defined in CorporateEntity
                company_dict = {k: v for k, v in company_dict_raw.items() if k in CorporateEntity.model_fields}

                # Handle potential empty strings for optional numeric fields from CSV before Pydantic validation
                optional_numeric_fields = {
                    'revenue_usd_millions': float,
                    'management_quality_score': int,
                    # Add any other Optional[numeric] fields from CorporateEntity here
                }
                for num_field, field_type in optional_numeric_fields.items():
                    if num_field in company_dict and company_dict[num_field] == '':
                        company_dict[num_field] = None
                    # Optionally, try to convert if it's a non-empty string but Pydantic might need help
                    # elif num_field in company_dict and isinstance(company_dict[num_field], str) and company_dict[num_field]:
                    #     try:
                    #         company_dict[num_field] = field_type(company_dict[num_field])
                    #     except ValueError:
                    #         logger.warning(f"Could not convert value for {num_field} in company {company_dict.get('company_id')}, setting to None. Value: {company_dict[num_field]}")
                    #         company_dict[num_field] = None


                # Parse semicolon-separated fields into lists
                list_fields = ['subsidiaries', 'suppliers', 'customers', 'loan_agreement_ids', 'financial_statement_ids']
                for field in list_fields:
                    if company_dict.get(field) and isinstance(company_dict[field], str):
                        company_dict[field] = [item.strip() for item in company_dict[field].split(';') if item.strip()]
                    elif company_dict.get(field) is None or pd.isna(company_dict.get(field)): # Handle None or NaN
                        company_dict[field] = None # Or [] if appropriate for your model
                    elif not isinstance(company_dict[field], list) and company_dict[field] is not None: # if it's some other non-list, non-string type
                        logger.warning(f"Company {company_dict.get('company_id')} field {field} is not a string or list, setting to None. Value: {company_dict[field]}")
                        company_dict[field] = None


                # Ensure founded_date is date object or None
                if 'founded_date' in company_dict and company_dict['founded_date'] is not None:
                    try:
                        company_dict['founded_date'] = pd.to_datetime(company_dict['founded_date']).date()
                    except ValueError:
                        logger.warning(f"Could not parse founded_date for {company_dict.get('company_id')}: {company_dict['founded_date']}. Setting to None.")
                        company_dict['founded_date'] = None

                from pydantic import ValidationError # Local import for specific catch
                try:
                    validated_companies.append(CorporateEntity(**company_dict))
                except ValidationError as ve:
                    logger.error(f"PYDANTIC VALIDATION ERROR for company {company_dict.get('company_id')}: {ve.errors()}. Data: {company_dict}")
                except Exception as e:
                    logger.error(f"GENERAL ERROR validating company {company_dict.get('company_id')}: {e}. Data: {company_dict}")

            # For now, keep companies as a DataFrame of validated dicts for easier filtering with pandas syntax.
            # Alternatively, store a list of CorporateEntity objects.
            if validated_companies:
                self._companies_df = pd.DataFrame([company.model_dump() for company in validated_companies])
                # Restore datetime objects if model_dump converted them to strings
                if 'founded_date' in self._companies_df.columns:
                     self._companies_df['founded_date'] = pd.to_datetime(self._companies_df['founded_date'], errors='coerce').dt.date
            else:
                self._companies_df = pd.DataFrame(columns=CorporateEntity.model_fields.keys())

            logger.info(f"Loaded and validated {len(self._companies_df)} company records.")

        except FileNotFoundError:
            logger.error(f"Company data file not found at {self.companies_data_path}. No company data loaded.")
            self._companies_df = pd.DataFrame(columns=CorporateEntity.model_fields.keys())
        except Exception as e:
            logger.error(f"Error loading company data: {e}")
            self._companies_df = pd.DataFrame(columns=CorporateEntity.model_fields.keys())

        # Load Loans
        self._loans_data = []
        try:
            logger.info(f"Loading loan data from: {self.loans_data_path}")
            with open(self.loans_data_path, 'r') as f:
                raw_loans_data = json.load(f)

            validated_loans = []
            for loan_dict in raw_loans_data:
                try:
                    validated_loans.append(LoanAgreement(**loan_dict))
                except Exception as e:
                    logger.error(f"Validation error for loan {loan_dict.get('loan_id')}: {e}. Data: {loan_dict}")
            self._loans_data = validated_loans
            logger.info(f"Loaded and validated {len(self._loans_data)} loan records.")
        except FileNotFoundError:
            logger.error(f"Loan data file not found at {self.loans_data_path}. No loan data loaded.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.loans_data_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading or validating loan data: {e}")

        # Load Financial Statements
        self._financial_statements_data = []
        try:
            logger.info(f"Loading financial statements from: {self.financial_statements_path}")
            with open(self.financial_statements_path, 'r') as f:
                raw_fs_data = json.load(f)

            validated_fs = []
            for fs_dict in raw_fs_data:
                try:
                    validated_fs.append(FinancialStatement(**fs_dict))
                except Exception as e:
                    logger.error(f"Validation error for financial statement {fs_dict.get('statement_id')}: {e}. Data: {fs_dict}")
            self._financial_statements_data = validated_fs
            logger.info(f"Loaded and validated {len(self._financial_statements_data)} financial statement records.")
        except FileNotFoundError:
            logger.error(f"Financial statements file not found at {self.financial_statements_path}. No data loaded.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.financial_statements_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading or validating financial statement data: {e}")

        # Load Default Events
        self._default_events_data = []
        try:
            logger.info(f"Loading default events from: {self.default_events_path}")
            with open(self.default_events_path, 'r') as f:
                raw_de_data = json.load(f)

            validated_de = []
            for de_dict in raw_de_data:
                try:
                    validated_de.append(DefaultEvent(**de_dict))
                except Exception as e:
                    logger.error(f"Validation error for default event {de_dict.get('event_id')}: {e}. Data: {de_dict}")
            self._default_events_data = validated_de
            logger.info(f"Loaded and validated {len(self._default_events_data)} default event records.")
        except FileNotFoundError:
            logger.error(f"Default events file not found at {self.default_events_path}. No data loaded.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.default_events_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading or validating default event data: {e}")

    def get_company_profile(self, company_id: str) -> Optional[CorporateEntity]:
        """Retrieves a company profile by its ID."""
        if self._companies_df is None or self._companies_df.empty:
            logger.warning("Company data is not loaded.")
            return None

        company_series = self._companies_df[self._companies_df['company_id'] == company_id].iloc[0]
        if not company_series.empty:
            company_dict = company_series.where(pd.notnull(company_series), None).to_dict()
            # Ensure date fields are actual date objects if they were converted by pandas
            if 'founded_date' in company_dict and isinstance(company_dict['founded_date'], pd.Timestamp):
                if pd.isna(company_dict['founded_date']): # Handle NaT
                    company_dict['founded_date'] = None
                else:
                    company_dict['founded_date'] = company_dict['founded_date'].date()
            elif 'founded_date' in company_dict and company_dict['founded_date'] is None: # Already None
                pass # It's fine
            elif 'founded_date' in company_dict and not isinstance(company_dict['founded_date'], datetime.date):
                 logger.warning(f"founded_date for {company_id} is not a date object after DataFrame retrieval: {company_dict['founded_date']}. Attempting conversion or setting to None.")
                 try:
                     company_dict['founded_date'] = pd.to_datetime(company_dict['founded_date']).date()
                 except ValueError:
                     company_dict['founded_date'] = None


            try:
                return CorporateEntity(**company_dict)
            except Exception as e:
                logger.error(f"Error creating CorporateEntity for {company_id}: {e}. Data: {company_dict}")
                return None
        logger.warning(f"Company with ID {company_id} not found.")
        return None

    def get_all_companies(self, industry_sector: Optional[IndustrySector] = None, country_iso_code: Optional[str] = None) -> List[CorporateEntity]:
        """Retrieves all company profiles, with optional filtering."""
        if self._companies_df is None or self._companies_df.empty:
            logger.info("Company DataFrame is not loaded or is empty.")
            return []

        # Start with the full DataFrame
        filtered_df = self._companies_df.copy()

        if industry_sector:
            if 'industry_sector' in filtered_df.columns:
                # Assuming industry_sector in df is stored as string from CorporateEntity.model_dump()
                # and IndustrySector enum values are strings.
                filtered_df = filtered_df[filtered_df['industry_sector'] == industry_sector.value]
            else:
                 logger.warning("industry_sector column not found in companies DataFrame for filtering.")
                 # If the column essential for filtering is missing, return empty or handle as error
                 return []

        if country_iso_code:
            if 'country_iso_code' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['country_iso_code'].str.upper() == country_iso_code.upper()]
            else:
                logger.warning("country_iso_code column not found in companies DataFrame for filtering.")
                return []

        companies = []
        for _, row in filtered_df.iterrows():
            company_dict = row.where(pd.notnull(row), None).to_dict()
            # Ensure date fields are actual date objects if they were converted by pandas
            if 'founded_date' in company_dict and isinstance(company_dict['founded_date'], pd.Timestamp):
                if pd.isna(company_dict['founded_date']): # Handle NaT
                    company_dict['founded_date'] = None
                else:
                    company_dict['founded_date'] = company_dict['founded_date'].date()

            # Convert list-like fields from stringified lists back to lists if necessary
            # This step might be redundant if _load_data already stores them correctly as lists in the DataFrame
            # However, model_dump() from Pydantic might convert lists to strings if not handled.
            # The _load_data stores them as proper lists after parsing from CSV.
            # And model_dump() by default keeps them as lists. So this might not be strictly needed here
            # but good to be aware of. The current _load_data stores dicts from model_dump(), so types should be fine.

            try:
                companies.append(CorporateEntity(**company_dict))
            except Exception as e:
                logger.error(f"Error creating CorporateEntity from filtered row: {e}. Data: {company_dict}")

        logger.info(f"Returning {len(companies)} companies after filtering.")
        return companies

    def get_loans_for_company(self, company_id: str) -> List[LoanAgreement]:
        """Retrieves all loans associated with a specific company ID."""
        if not self._loans_data:
            return []

        company_loans = []
        for loan_obj in self._loans_data: # Iterate over LoanAgreement objects
            if loan_obj.company_id == company_id: # Access attributes directly
                company_loans.append(loan_obj)
        return company_loans

    def get_all_loans(self) -> List[LoanAgreement]:
        """Retrieves all loan records."""
        if not self._loans_data:
            return []
        # self._loans_data already contains LoanAgreement objects
        return self._loans_data

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

    def get_financial_statements_for_company(self, company_id: str) -> List[FinancialStatement]:
        """Retrieves all financial statements for a specific company ID."""
        if not self._financial_statements_data:
            return []
        return [fs for fs in self._financial_statements_data if fs.company_id == company_id]

    def get_default_events_for_loan(self, loan_id: str) -> List[DefaultEvent]:
        """Retrieves all default events for a specific loan ID."""
        if not self._default_events_data:
            return []
        return [de for de in self._default_events_data if de.loan_id == loan_id]

    def get_loans_by_criteria(
        self,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        currency: Optional[Currency] = None,
        collateral_type: Optional[CollateralType] = None,
        default_status: Optional[bool] = None
    ) -> List[LoanAgreement]:
        """Retrieves loans based on multiple optional criteria."""
        if not self._loans_data:
            return []

        filtered_loans = self._loans_data

        if min_amount is not None:
            filtered_loans = [loan for loan in filtered_loans if loan.loan_amount >= min_amount]
        if max_amount is not None:
            filtered_loans = [loan for loan in filtered_loans if loan.loan_amount <= max_amount]
        if currency is not None:
            filtered_loans = [loan for loan in filtered_loans if loan.currency == currency]
        if collateral_type is not None:
            filtered_loans = [loan for loan in filtered_loans if loan.collateral_type == collateral_type]
        if default_status is not None:
            filtered_loans = [loan for loan in filtered_loans if loan.default_status == default_status]

        return filtered_loans

if __name__ == "__main__":
    # This requires src.core.logging_config to be importable,
    # which means running from the project root with `python -m src.data_management.knowledge_base`
    # or ensuring PYTHONPATH is set up correctly.

    # Setup basic logging for the __main__ block if not configured globally
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Assuming running from project root for correct paths:
    # from src.core.logging_config import setup_logging
    # setup_logging()

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
