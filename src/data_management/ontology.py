from enum import Enum
from typing import List, Optional, Any, Dict
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

class Counterparty(BaseModel):
    counterparty_id: str
    name: str
    type: str  # e.g., "Bank", "Supplier", "Customer", "Guarantor"
    details: Optional[Dict[str, Any]] = None

class FinancialInstrument(BaseModel):
    instrument_id: str
    instrument_type: str  # e.g., "Loan", "Bond", "Equity"
    issuer_id: Optional[str] = None  # e.g., company_id for bonds/equity
    details: Optional[Dict[str, Any]] = None  # e.g., coupon_rate for bonds, maturity_date

class MarketDataPoint(BaseModel):
    data_point_id: str
    data_type: str  # e.g., "InterestRateBenchmark", "StockIndex", "CommodityPrice", "FXRate"
    value: float
    timestamp: datetime.datetime
    source: Optional[str] = None
    instrument_id_or_identifier: Optional[str] = None  # e.g., specific stock ticker or bond ISIN

class CorporateEntity(BaseModel):
    company_id: str = Field(..., description="Unique identifier for the corporate entity")
    company_name: str = Field(..., description="Legal name of the company")
    industry_sector: IndustrySector = Field(..., description="Primary industry sector of operation")
    country_iso_code: str = Field(min_length=2, max_length=3, description="ISO 3166-1 alpha-2 or alpha-3 country code")
    founded_date: Optional[datetime.date] = None
    revenue_usd_millions: Optional[float] = Field(None, ge=0)
    subsidiaries: Optional[List[str]] = Field(None, description="List of company_ids of subsidiaries")
    suppliers: Optional[List[str]] = Field(None, description="List of company_ids of key suppliers")
    customers: Optional[List[str]] = Field(None, description="List of company_ids of key customers")
    loan_agreement_ids: Optional[List[str]] = Field(None, description="List of loan_ids associated with this company")
    financial_statement_ids: Optional[List[str]] = Field(None, description="List of statement_ids associated with this company")

    management_quality_score: Optional[int] = Field(None, ge=1, le=10, description="Subjective score of management quality (1-10)")
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
    guarantors: Optional[List[str]] = Field(None, description="List of company_ids acting as guarantors")
    syndicate_members: Optional[List[str]] = Field(None, description="List of counterparty_ids (banks) in a syndicated loan")
    security_details: Optional[str] = None

    seniority_of_debt: Optional[str] = Field(None, description="Seniority level of the debt (e.g., Senior, Subordinated, Secured, Unsecured)")
    economic_condition_indicator: Optional[float] = Field(None, ge=0, le=1, description="Synthetic indicator of economic conditions at loan origination or at a point in time (0-1, higher is better)")

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
    currency: Currency
    reporting_period_months: int
    revenue: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    ebitda: Optional[float] = None
    interest_expense: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    income_before_tax: Optional[float] = None
    tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    accounts_receivable: Optional[float] = None
    inventory: Optional[float] = None
    current_assets: Optional[float] = None
    property_plant_equipment_net: Optional[float] = None
    total_non_current_assets: Optional[float] = None
    accounts_payable: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    total_non_current_liabilities: Optional[float] = None
    retained_earnings: Optional[float] = None
    equity_attributable_to_parent: Optional[float] = None
    cash_flow_operations: Optional[float] = None
    cash_flow_investing: Optional[float] = None
    cash_flow_financing: Optional[float] = None
    capital_expenditures: Optional[float] = None
    # ... other financial metrics

class DefaultEvent(BaseModel):
    event_id: str
    loan_id: str
    company_id: str
    default_date: datetime.date
    reason: Optional[str] = None
    default_type: Optional[str] = None  # e.g., "PaymentMissed", "CovenantBreach", "Bankruptcy"
    exposure_at_default_usd: Optional[float] = None
    recovery_amount_usd: Optional[float] = None
    loss_given_default_actual: Optional[float] = Field(None, description="Actual LGD, calculated as (EAD - Recovery) / EAD if applicable")


class RiskItem(BaseModel):
    """
    Represents a single item in the risk probability map, typically a loan,
    augmented with company information, risk scores, and other relevant metrics.
    """
    # Identifiers
    loan_id: str
    company_id: str

    # Company Information
    company_name: str
    industry_sector: Optional[str] = None # Using str representation of IndustrySector enum for flexibility
    country_iso_code: Optional[str] = None

    # Loan Information
    loan_amount_usd: float
    currency: str # Using str representation of Currency enum
    collateral_type: Optional[str] = None # Using str representation of CollateralType enum
    is_defaulted: bool
    origination_date: Optional[datetime.date] = None
    maturity_date: Optional[datetime.date] = None


    # Core Risk Metrics
    pd_estimate: Optional[float] = None
    lgd_estimate: Optional[float] = None
    exposure_at_default_usd: Optional[float] = None # Typically loan_amount for this PoC
    expected_loss_usd: Optional[float] = None

    # Qualitative Scores (Original and HITL-augmented)
    management_quality_score: Optional[int] = None # Original score from data
    hitl_management_quality_score: Optional[int] = None # Score after potential HITL review/override

    # KG-Derived Metrics
    kg_degree_centrality: Optional[float] = None
    kg_num_suppliers: Optional[int] = None
    kg_num_customers: Optional[int] = None
    kg_num_subsidiaries: Optional[int] = None

    # HITL Annotations
    hitl_review_status: Optional[str] = None # e.g., "Pending Review", "Reviewed", "Flagged"
    hitl_analyst_notes: Optional[str] = None # Textual notes from an analyst
    hitl_pd_override: Optional[float] = None # Analyst overridden PD
    hitl_lgd_override: Optional[float] = None # Analyst overridden LGD

    # Timestamps / Versioning
    data_as_of_date: Optional[datetime.date] = Field(default_factory=datetime.date.today)
    risk_calculation_timestamp: Optional[datetime.datetime] = Field(default_factory=datetime.datetime.now)

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


class HITLReviewStatus(str, Enum):
    PENDING_REVIEW = "Pending Analyst Review"
    REVIEWED_OK = "Reviewed - OK"
    FLAGGED_ATTENTION = "Flagged - Needs Attention"
    FLAGGED_MODEL_DISAGREEMENT = "Flagged - Model Disagreement"
    OVERRIDDEN = "Overridden by Analyst"

class HITLAnnotation(BaseModel):
    """
    Represents Human-in-the-Loop annotations for a specific entity (e.g., company or loan).
    Using entity_id to be generic, can be company_id or loan_id based on context.
    Multiple annotations can exist for the same entity (e.g., from different analysts or at different times).
    """
    annotation_id: str = Field(default_factory=lambda: f"hitl_anno_{datetime.datetime.now().timestamp()}_{hash(str(datetime.datetime.now()))}", description="Unique identifier for the annotation itself")
    entity_id: str = Field(..., description="Identifier for the entity being annotated (e.g., company_id or loan_id)")
    annotation_target_field: Optional[str] = Field(None, description="Specific field being annotated/overridden, e.g., 'pd_estimate', 'management_quality_score'")
    annotation_type: str = Field(..., description="Type of entity being annotated, e.g., 'company', 'loan', 'risk_item_summary'")

    # Specific HITL fields - these are now more generic, specific values stored in `new_value` or `notes`
    hitl_review_status: Optional[HITLReviewStatus] = Field(None, description="Review status set by analyst")
    hitl_analyst_notes: Optional[str] = Field(None, description="Textual notes or justifications from the analyst")

    # Fields for overrides or suggested values
    previous_value_numeric: Optional[float] = Field(None, description="Previous numeric value of the target field, if applicable")
    new_value_numeric: Optional[float] = Field(None, description="New numeric value suggested/overridden by analyst")
    previous_value_str: Optional[str] = Field(None, description="Previous string value of the target field, if applicable")
    new_value_str: Optional[str] = Field(None, description="New string value suggested/overridden by analyst")

    override_confidence: Optional[float] = Field(None, ge=0, le=1, description="Analyst's confidence in the override (0-1)")
    reason_code: Optional[str] = Field(None, description="Predefined reason code for the annotation or override")

    # Metadata for the annotation itself
    annotator_id: Optional[str] = Field(None, description="Identifier of the analyst who made the annotation")
    annotation_timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of when the annotation was last made/updated")
    version: int = Field(1, description="Version of this annotation for the entity_id/target_field, if multiple edits occur")

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }

# Update RiskItem to use the new HITLReviewStatus and reflect more detailed HITL integration
class RiskItem(BaseModel):
    """
    Represents a single item in the risk probability map, typically a loan,
    augmented with company information, risk scores, and other relevant metrics.
    """
    # Identifiers
    loan_id: str
    company_id: str

    # Company Information
    company_name: str
    industry_sector: Optional[str] = None
    country_iso_code: Optional[str] = None
    founded_date: Optional[datetime.date] = None # Added from CorporateEntity

    # Loan Information
    loan_amount_usd: float
    currency: str
    collateral_type: Optional[str] = None
    collateral_value_usd: Optional[float] = None # Added from LoanAgreement
    is_defaulted: bool
    origination_date: Optional[datetime.date] = None
    maturity_date: Optional[datetime.date] = None
    interest_rate_percentage: Optional[float] = None # Added from LoanAgreement
    seniority_of_debt: Optional[str] = None # Added from LoanAgreement
    economic_condition_indicator: Optional[float] = None # Added from LoanAgreement


    # Core Risk Metrics
    model_pd_estimate: Optional[float] = Field(None, description="PD estimate from the model")
    model_lgd_estimate: Optional[float] = Field(None, description="LGD estimate from the model")
    model_expected_loss_usd: Optional[float] = Field(None, description="EL calculated from model PD & LGD")

    # Effective Risk Metrics (after potential HITL overrides)
    effective_pd_estimate: Optional[float] = Field(None, description="Effective PD estimate (model or HITL override)")
    effective_lgd_estimate: Optional[float] = Field(None, description="Effective LGD estimate (model or HITL override)")
    effective_expected_loss_usd: Optional[float] = Field(None, description="EL calculated from effective PD & LGD")

    exposure_at_default_usd: Optional[float] = None

    # Qualitative Scores
    original_management_quality_score: Optional[int] = Field(None, description="Original management quality score from data source")
    effective_management_quality_score: Optional[int] = Field(None, description="Effective MQS (original or HITL override)")

    # KG-Derived Metrics
    kg_degree_centrality: Optional[float] = None
    kg_num_suppliers: Optional[int] = None
    kg_num_customers: Optional[int] = None
    kg_num_subsidiaries: Optional[int] = None

    # HITL Summary/Status fields directly on RiskItem for convenience
    hitl_overall_review_status: Optional[HITLReviewStatus] = Field(None, description="Overall review status based on latest HITLAnnotation for this loan/company")
    hitl_last_annotation_timestamp: Optional[datetime.datetime] = Field(None, description="Timestamp of the most recent HITL annotation relevant to this item")
    hitl_has_notes: bool = Field(False, description="True if there are any analyst notes")
    hitl_has_pd_override: bool = Field(False, description="True if PD has been overridden by an analyst")
    hitl_has_lgd_override: bool = Field(False, description="True if LGD has been overridden by an analyst")
    hitl_has_mqs_override: bool = Field(False, description="True if MQS has been overridden by an analyst")


    # Trend Indicators / Peer Comparisons (Conceptual placeholders)
    pd_trend_3m: Optional[str] = Field(None, description="3-month trend of PD (e.g., 'UP', 'DOWN', 'STABLE')")
    el_peer_percentile: Optional[float] = Field(None, ge=0, le=100, description="Percentile of this item's EL compared to industry peers")

    # Timestamps / Versioning
    data_as_of_date: Optional[datetime.date] = Field(default_factory=datetime.date.today)
    risk_calculation_timestamp: Optional[datetime.datetime] = Field(default_factory=datetime.datetime.now)

    model_config = {
        "validate_assignment": True,
        "extra": "forbid"
    }


if __name__ == "__main__":
    # Example Usage
    sample_company = CorporateEntity(
        company_id="COMP001",
        company_name="Innovatech Solutions",
        industry_sector=IndustrySector.TECHNOLOGY,
        country_iso_code="USA",
        founded_date=datetime.date(2010, 5, 15),
        revenue_usd_millions=150.75,
        loan_agreement_ids=["LOAN7001"]
    )
    print(f"Sample Company: {sample_company.company_name}, Sector: {sample_company.industry_sector.value}, Loan IDs: {sample_company.loan_agreement_ids}")

    sample_loan = LoanAgreement(
        loan_id="LOAN7001",
        company_id="COMP001",
        loan_amount=5000000,
        currency=Currency.USD,
        origination_date=datetime.date(2022, 8, 15),
        maturity_date=datetime.date(2027, 8, 15),
        interest_rate_percentage=5.5,
        collateral_type=CollateralType.REAL_ESTATE,
        guarantors=["COMP002", "COMP003"]
    )
    print(f"Sample Loan: {sample_loan.loan_id} for {sample_loan.company_id}, Amount: {sample_loan.loan_amount} {sample_loan.currency.value}, Guarantors: {sample_loan.guarantors}")

    sample_financial_statement = FinancialStatement(
        statement_id="FS2023Q4",
        company_id="COMP001",
        statement_date=datetime.date(2023, 12, 31),
        total_assets_usd=75000000,
        total_liabilities_usd=30000000,
        net_equity_usd=45000000,
        currency=Currency.USD,
        reporting_period_months=12,
        revenue=50000000,
        net_income=5000000,
        cash_flow_operations=7000000
    )
    print(f"Sample Financial Statement: {sample_financial_statement.statement_id} for {sample_financial_statement.company_id}, Net Income: {sample_financial_statement.net_income}")

    sample_default_event = DefaultEvent(
        event_id="DEF001",
        loan_id="LOAN7002", # Assuming another loan for this example
        company_id="COMP004", # Assuming another company for this example
        default_date=datetime.date(2024, 1, 10),
        default_type="PaymentMissed",
        exposure_at_default_usd=1200000,
        recovery_amount_usd=300000,
        loss_given_default_actual=0.75
    )
    print(f"Sample Default Event: {sample_default_event.event_id} for loan {sample_default_event.loan_id}, Type: {sample_default_event.default_type}, LGD: {sample_default_event.loss_given_default_actual}")

    try:
        invalid_loan = LoanAgreement(
            loan_id="LOAN_ERR", company_id="C_ERR", loan_amount=100, currency=Currency.USD,
            origination_date=datetime.date(2023,1,1), maturity_date=datetime.date(2022,1,1), # Invalid dates
            interest_rate_percentage=5.0
        )
    except ValueError as e:
        print(f"Error creating loan: {e}")
