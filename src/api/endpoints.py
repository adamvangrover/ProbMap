import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from typing import List, Dict, Any
import datetime # Added for temporary ID generation

from src.api import schemas # Import Pydantic models from schemas.py
from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import CorporateEntity as OntologyCorporateEntity, LoanAgreement as OntologyLoanAgreement
from src.data_management.ontology import IndustrySector, Currency, CollateralType # Import enums for defaults
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.risk_models.pricing_model import PricingModel
# from src.risk_map.risk_map_service import RiskMapService # For future endpoints like /risk-map-overview

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Initialize Services (Dependencies) ---
# These will be treated somewhat like singletons for the PoC API.
# In a more complex app, you might use FastAPI's Depends for dependency injection.

# Initialize KnowledgeBaseService
kb_service = KnowledgeBaseService() # Loads data on init

# Initialize Models - attempt to load pre-trained, or train if not found (for PoC convenience)
pd_model_instance = PDModel()
if not pd_model_instance.load_model():
    logger.warning("API: PD model file not found or failed to load. Attempting to train (PoC behavior)...")
    if kb_service.get_all_loans() and kb_service.get_all_companies():
        metrics = pd_model_instance.train(kb_service)
        if "error" in metrics:
            logger.error(f"API: Failed to train PD model: {metrics['error']}. PD predictions will be unavailable/defaulted.")
        else:
            logger.info(f"API: PD model trained with metrics: {metrics}")
    else:
        logger.error("API: Cannot train PD model as KnowledgeBase has no data.")

lgd_model_instance = LGDModel()
if not lgd_model_instance.load_model():
    logger.warning("API: LGD model file not found or failed to load. Attempting to train (PoC behavior)...")
    if kb_service.get_all_loans(): # LGD model training relies on loans (specifically defaulted ones)
        metrics = lgd_model_instance.train(kb_service)
        if "error" in metrics:
            logger.error(f"API: Failed to train LGD model: {metrics['error']}. LGD predictions will be unavailable/defaulted.")
        else:
            logger.info(f"API: LGD model trained with metrics: {metrics}")
    else:
        logger.error("API: Cannot train LGD model as KnowledgeBase has no loan data.")

pricing_model_instance = PricingModel()

# --- Helper function to get or construct company/loan data for models ---
def _get_model_input_data(company_input: schemas.CompanyInput, loan_input: schemas.LoanInput, kb: KnowledgeBaseService) -> Dict[str, Any]:
    warnings = []
    # Company data: Try to fetch from KB, else use provided input
    company_data_for_model: Dict[str, Any] = {}
    if company_input.company_id:
        company_from_kb = kb.get_company_profile(company_input.company_id)
        if company_from_kb:
            company_data_for_model = company_from_kb.model_dump()
            logger.info(f"Using company {company_input.company_id} from KB for model input.")
        else:
            warnings.append(f"Company ID {company_input.company_id} not found in KB. Using provided info if any.")
            # Fallback to use data from CompanyInput directly if KB lookup fails
            company_data_for_model = company_input.model_dump(exclude_none=True)
            # Ensure company_id from path is retained if it was the basis of a failed lookup
            company_data_for_model['company_id'] = company_input.company_id

    else: # No company_id provided in CompanyInput (should be caught by Pydantic if company_id is non-optional in schema)
          # This path might be less likely if company_id is always required by CompanyInput schema.
        company_data_for_model = company_input.model_dump(exclude_none=True)
        # If company_id somehow still missing (e.g. if schema made it optional and it was None)
        if not company_data_for_model.get('company_id'):
            company_data_for_model['company_id'] = "TEMP_COMPANY_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            warnings.append(f"No company_id was provided in CompanyInput, using temporary ID: {company_data_for_model['company_id']}")


    # Loan data: Primarily use provided input, as it might be a new/hypothetical loan
    loan_data_for_model: Dict[str, Any] = loan_input.model_dump(exclude_none=True)
    if not loan_data_for_model.get('loan_id'):
        loan_data_for_model['loan_id'] = "TEMP_LOAN_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        warnings.append(f"No loan_id provided, using temporary ID: {loan_data_for_model['loan_id']}")

    # Ensure required fields for models are present with defaults if missing
    # This depends on what PDModel.predict_for_loan and LGDModel.predict_lgd expect

    # For company_data_for_model (ensure critical fields from CompanyInput are there)
    # Convert enums to their values if they are enums, or use defaults
    if 'industry_sector' not in company_data_for_model or company_data_for_model.get('industry_sector') is None:
        company_data_for_model['industry_sector'] = IndustrySector.OTHER.value
    elif isinstance(company_data_for_model.get('industry_sector'), IndustrySector):
        company_data_for_model['industry_sector'] = company_data_for_model['industry_sector'].value

    if 'country_iso_code' not in company_data_for_model or company_data_for_model.get('country_iso_code') is None:
        company_data_for_model['country_iso_code'] = "USA" # Default
    if 'revenue_usd_millions' not in company_data_for_model or company_data_for_model.get('revenue_usd_millions') is None:
        company_data_for_model['revenue_usd_millions'] = 0 # Default
    # founded_date can be None

    # For loan_data_for_model (ensure critical fields from LoanInput are there)
    if 'loan_amount' not in loan_data_for_model: # Should be caught by Pydantic if not optional
        loan_data_for_model['loan_amount'] = 0

    if 'currency' not in loan_data_for_model or loan_data_for_model.get('currency') is None:
        loan_data_for_model['currency'] = Currency.USD.value
    elif isinstance(loan_data_for_model.get('currency'), Currency):
        loan_data_for_model['currency'] = loan_data_for_model['currency'].value

    if 'collateral_type' not in loan_data_for_model or loan_data_for_model.get('collateral_type') is None:
        loan_data_for_model['collateral_type'] = CollateralType.NONE.value
    elif isinstance(loan_data_for_model.get('collateral_type'), CollateralType):
        loan_data_for_model['collateral_type'] = loan_data_for_model['collateral_type'].value

    if 'interest_rate_percentage' not in loan_data_for_model or loan_data_for_model.get('interest_rate_percentage') is None:
        loan_data_for_model['interest_rate_percentage'] = 5.0 # Default


    return {
        "company_dict": company_data_for_model,
        "loan_dict": loan_data_for_model,
        "warnings": warnings
    }

# --- API Endpoints ---

@router.get("/company/{company_id}", response_model=schemas.CompanyResponse)
async def get_company_info(company_id: str):
    logger.info(f"API: Received request for company info: {company_id}")
    company = kb_service.get_company_profile(company_id)
    if not company:
        logger.warning(f"API: Company {company_id} not found in Knowledge Base.")
        raise HTTPException(status_code=404, detail=f"Company with ID {company_id} not found.")
    # Convert OntologyCorporateEntity to schemas.CompanyResponse
    return schemas.CompanyResponse(**company.model_dump())


@router.post("/calculate-risk-metrics", response_model=schemas.RiskMetricsResponse)
async def calculate_risk_metrics(request: schemas.RiskMetricsRequest = Body(...)):
    logger.info(f"API: Received request to calculate risk metrics for company: {request.company_info.company_id}")

    input_data = _get_model_input_data(request.company_info, request.loan_info, kb_service)
    company_dict = input_data["company_dict"]
    loan_dict = input_data["loan_dict"]
    warnings = input_data["warnings"]

    # PD Calculation
    pd_pred_class, pd_prob = None, 0.5 # Default if model fails
    if pd_model_instance.model:
        pd_result = pd_model_instance.predict_for_loan(loan_dict, company_dict)
        if pd_result:
            pd_pred_class, pd_prob = pd_result
        else:
            warnings.append("PD prediction failed; using default PD.")
    else:
        warnings.append("PD model not available; using default PD.")

    # LGD Calculation
    lgd_val = 0.75 # Default if model fails
    if lgd_model_instance.model:
        # LGD model expects features like 'collateral_type', 'loan_amount_usd'
        lgd_features = {
            'collateral_type': loan_dict.get('collateral_type', CollateralType.NONE.value),
            'loan_amount_usd': loan_dict.get('loan_amount', 0)
        }
        lgd_val = lgd_model_instance.predict_lgd(lgd_features)
    else:
        warnings.append("LGD model not available; using default LGD.")

    # Expected Loss
    exposure = loan_dict.get('loan_amount', 0)
    expected_loss = pd_prob * lgd_val * exposure

    return schemas.RiskMetricsResponse(
        request_details=request,
        company_id=company_dict.get('company_id', "N/A"),
        loan_id=loan_dict.get('loan_id', "N/A"),
        pd_estimate=pd_prob,
        lgd_estimate=lgd_val,
        pd_model_prediction_class=pd_pred_class,
        expected_loss_estimate=expected_loss,
        exposure_at_default=exposure,
        warnings=warnings
    )

@router.post("/price-loan", response_model=schemas.LoanPricingResponse)
async def price_loan(request: schemas.LoanPricingRequest = Body(...)):
    logger.info(f"API: Received request to price loan for company: {request.company_info.company_id}")

    input_data = _get_model_input_data(request.company_info, request.loan_info, kb_service)
    company_dict_for_models = input_data["company_dict"]
    loan_dict_for_models = input_data["loan_dict"]
    warnings = input_data["warnings"]

    # Determine PD and LGD to use
    pd_to_use = request.manual_pd_estimate
    lgd_to_use = request.manual_lgd_estimate

    if pd_to_use is None:
        if pd_model_instance.model:
            pd_result = pd_model_instance.predict_for_loan(loan_dict_for_models, company_dict_for_models)
            if pd_result:
                _, pd_to_use = pd_result
            else:
                pd_to_use = 0.5 # Default
                warnings.append("PD prediction failed; using default PD for pricing.")
        else:
            pd_to_use = 0.5 # Default
            warnings.append("PD model not available; using default PD for pricing.")

    if lgd_to_use is None:
        if lgd_model_instance.model:
            lgd_features = {
                'collateral_type': loan_dict_for_models.get('collateral_type', CollateralType.NONE.value),
                'loan_amount_usd': loan_dict_for_models.get('loan_amount', 0)
            }
            lgd_to_use = lgd_model_instance.predict_lgd(lgd_features)
        else:
            lgd_to_use = 0.75 # Default
            warnings.append("LGD model not available; using default LGD for pricing.")

    # Get pricing
    # The pricing model's calculate_price takes company_data and loan_data dicts.
    pricing_result = pricing_model_instance.calculate_price(
        pd_estimate=pd_to_use,
        lgd_estimate=lgd_to_use,
        company_data=company_dict_for_models,
        loan_data=loan_dict_for_models
    )

    return schemas.LoanPricingResponse(
        request_details=request,
        suggested_interest_rate=pricing_result["suggested_interest_rate"],
        pd_used=pd_to_use,
        lgd_used=lgd_to_use,
        pricing_components=pricing_result,
        warnings=warnings
    )

# Example placeholder for a Risk Map related endpoint (conceptual)
# @router.get("/portfolio-risk-overview", response_model=List[Dict[str, Any]]) # Define a Pydantic model for this
# async def get_portfolio_overview():
#     # risk_map_serv = RiskMapService(kb_service, pd_model_instance, lgd_model_instance)
#     # overview = risk_map_serv.generate_portfolio_risk_overview()
#     # return overview
#     logger.info("API: /portfolio-risk-overview endpoint called (placeholder).")
#     return [{"message": "Portfolio risk overview endpoint is conceptual for this PoC."}]
