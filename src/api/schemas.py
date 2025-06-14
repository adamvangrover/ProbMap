from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
import datetime

from src.data_management.ontology import IndustrySector, Currency, CollateralType # Reuse existing enums

# --- Request Models ---

class CompanyInput(BaseModel):
    company_id: str = Field(..., description="Unique identifier for the company, used for lookup if existing.")
    company_name: Optional[str] = None
    industry_sector: Optional[IndustrySector] = None
    country_iso_code: Optional[str] = Field(None, min_length=2, max_length=3)
    founded_date: Optional[datetime.date] = None
    revenue_usd_millions: Optional[float] = Field(None, ge=0)
    # Add other raw features the PD model might need if not looking up existing company
    # This model is flexible: can be used for lookup or for providing ad-hoc company data.

class LoanInput(BaseModel):
    loan_id: Optional[str] = Field(None, description="Unique identifier for the loan, if any.")
    loan_amount: float = Field(..., gt=0, description="Principal amount of the loan")
    currency: Currency = Currency.USD
    origination_date: Optional[datetime.date] = None # Will default if not provided
    maturity_date: Optional[datetime.date] = None # Will default if not provided
    interest_rate_percentage: Optional[float] = Field(None, ge=0, description="Current or proposed annual interest rate")
    collateral_type: CollateralType = CollateralType.NONE
    collateral_value_usd: Optional[float] = Field(None, ge=0)
    # Add other raw features the PD/LGD model might need

class RiskMetricsRequest(BaseModel):
    company_info: CompanyInput
    loan_info: LoanInput

class LoanPricingRequest(BaseModel):
    company_info: CompanyInput # Can be just company_id if company exists in KB
    loan_info: LoanInput       # Details of the loan to be priced
    # Optionally, allow manual override of PD/LGD if known
    manual_pd_estimate: Optional[float] = Field(None, ge=0, le=1)
    manual_lgd_estimate: Optional[float] = Field(None, ge=0, le=1)


# --- Response Models ---

class CompanyResponse(BaseModel):
    company_id: str
    company_name: str
    industry_sector: IndustrySector
    country_iso_code: str
    founded_date: Optional[datetime.date] = None
    revenue_usd_millions: Optional[float] = None
    # Include other relevant details from src.data_management.ontology.CorporateEntity

    class Config:
        orm_mode = True # For compatibility if creating from ORM objects / Pydantic models

class LoanResponse(BaseModel):
    loan_id: str
    company_id: str
    loan_amount: float
    currency: Currency
    # ... other fields from src.data_management.ontology.LoanAgreement

    class Config:
        orm_mode = True

class RiskMetricsResponse(BaseModel):
    request_details: RiskMetricsRequest
    company_id: str
    loan_id: Optional[str] = "N/A" # Loan might not have an ID yet if new
    pd_estimate: float = Field(..., description="Probability of Default (0.0 to 1.0)")
    lgd_estimate: float = Field(..., description="Loss Given Default (0.0 to 1.0)")
    pd_model_prediction_class: Optional[int] = None # 0 or 1
    expected_loss_estimate: Optional[float] = Field(None, description="Expected Loss (PD * LGD * Exposure)")
    exposure_at_default: float
    warnings: List[str] = []

class LoanPricingResponse(BaseModel):
    request_details: LoanPricingRequest
    suggested_interest_rate: float = Field(..., description="Suggested annual interest rate (%)")
    pd_used: float
    lgd_used: float
    pricing_components: Dict[str, Any] # To show breakdown from PricingModel
    warnings: List[str] = []

class MessageResponse(BaseModel):
    message: str
