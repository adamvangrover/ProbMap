from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator
import datetime

class IndustrySector(str, Enum):
    TECHNOLOGY = "Technology"
    CONSTRUCTION = "Construction"
    PHARMACEUTICALS = "Pharmaceuticals"
    LOGISTICS = "Logistics"
    AGRICULTURE = "Agriculture"
    FINANCIAL_SERVICES = "Financial Services"
    MANUFACTURING = "Manufacturing"
    OTHER = "Other"

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"

class CollateralType(str, Enum):
    REAL_ESTATE = "Real Estate"
    EQUIPMENT = "Equipment"
    RECEIVABLES = "Receivables"
    INVENTORY = "Inventory"
    INTELLECTUAL_PROPERTY = "Intellectual Property"
    NONE = "None"

class CorporateEntity(BaseModel):
    company_id: str = Field(..., description="Unique identifier for the corporate entity")
    company_name: str = Field(..., description="Legal name of the company")
    industry_sector: IndustrySector = Field(..., description="Primary industry sector of operation")
    country_iso_code: str = Field(min_length=2, max_length=3, description="ISO 3166-1 alpha-2 or alpha-3 country code")
    founded_date: Optional[datetime.date] = None
    revenue_usd_millions: Optional[float] = Field(None, ge=0)
    # Add other relevant fields like address, number of employees, etc.

class LoanAgreement(BaseModel):
    loan_id: str = Field(..., description="Unique identifier for the loan agreement")
    company_id: str = Field(..., description="Identifier of the borrowing company")
    loan_amount: float = Field(..., gt=0, description="Principal amount of the loan")
    currency: Currency = Field(..., description="Currency of the loan")
    origination_date: datetime.date
    maturity_date: datetime.date
    interest_rate_percentage: float = Field(..., ge=0, description="Annual interest rate")
    collateral_type: CollateralType = CollateralType.NONE
    collateral_value_usd: Optional[float] = Field(None, ge=0)
    default_status: bool = False

    @validator('maturity_date')
    def maturity_must_be_after_origination(cls, v, values):
        if 'origination_date' in values and v <= values['origination_date']:
            raise ValueError('Maturity date must be after origination date')
        return v

class FinancialStatement(BaseModel):
    statement_id: str
    company_id: str
    statement_date: datetime.date
    total_assets_usd: float
    total_liabilities_usd: float
    net_equity_usd: float
    # ... other financial metrics

class DefaultEvent(BaseModel):
    event_id: str
    loan_id: str
    company_id: str
    default_date: datetime.date
    reason: Optional[str] = None

if __name__ == "__main__":
    # Example Usage
    sample_company = CorporateEntity(
        company_id="COMP001",
        company_name="Innovatech Solutions",
        industry_sector=IndustrySector.TECHNOLOGY,
        country_iso_code="USA",
        founded_date=datetime.date(2010, 5, 15),
        revenue_usd_millions=150.75
    )
    print(f"Sample Company: {sample_company.company_name}, Sector: {sample_company.industry_sector.value}")

    sample_loan = LoanAgreement(
        loan_id="LOAN7001",
        company_id="COMP001",
        loan_amount=5000000,
        currency=Currency.USD,
        origination_date=datetime.date(2022, 8, 15),
        maturity_date=datetime.date(2027, 8, 15),
        interest_rate_percentage=5.5,
        collateral_type=CollateralType.REAL_ESTATE
    )
    print(f"Sample Loan: {sample_loan.loan_id} for {sample_loan.company_id}, Amount: {sample_loan.loan_amount} {sample_loan.currency.value}")
    try:
        invalid_loan = LoanAgreement(
            loan_id="LOAN_ERR", company_id="C_ERR", loan_amount=100, currency=Currency.USD,
            origination_date=datetime.date(2023,1,1), maturity_date=datetime.date(2022,1,1), # Invalid dates
            interest_rate_percentage=5.0
        )
    except ValueError as e:
        print(f"Error creating loan: {e}")
