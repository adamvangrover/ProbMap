import logging
from typing import List, Dict, Any, Optional

from src.data_management.knowledge_base import KnowledgeBaseService
from src.data_management.ontology import LoanAgreement, CorporateEntity, RiskItem, HITLAnnotation, HITLReviewStatus # Updated imports
from src.risk_models.pd_model import PDModel
from src.risk_models.lgd_model import LGDModel
from src.data_management.knowledge_graph import KnowledgeGraphService
import datetime # Ensure datetime is imported

import json # For pretty printing in __main__

logger = logging.getLogger(__name__)

class RiskMapService:
    def __init__(self,
                 kb_service: KnowledgeBaseService,
                 pd_model: PDModel,
                 lgd_model: LGDModel,
                 kg_service: Optional[KnowledgeGraphService] = None):
        self.kb_service = kb_service
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.kg_service = kg_service

        if self.pd_model.model is None:
            logger.info("PD model not loaded, attempting to load from default path.")
            if not self.pd_model.load_model():
                logger.warning("PD model could not be loaded. Risk map generation might be impaired or use defaults.")
        if self.lgd_model.model is None:
            logger.info("LGD model not loaded, attempting to load from default path.")
            if not self.lgd_model.load_model():
                logger.warning("LGD model could not be loaded. Risk map generation might be impaired or use defaults.")

    def generate_portfolio_risk_overview(
        self,
        industry_sector: Optional[str] = None,
        country_iso_code: Optional[str] = None,
        min_loan_amount_usd: Optional[float] = None,
        max_loan_amount_usd: Optional[float] = None,
        min_effective_pd_estimate: Optional[float] = None,
        max_effective_pd_estimate: Optional[float] = None,
        default_status: Optional[bool] = None,
        hitl_overall_review_status: Optional[HITLReviewStatus] = None,
        min_effective_management_quality_score: Optional[int] = None,
        max_effective_management_quality_score: Optional[int] = None
    ) -> List[RiskItem]:
        log_message = "Generating portfolio risk overview"
        filters_applied = []
        if industry_sector: filters_applied.append(f"industry_sector='{industry_sector}'")
        if country_iso_code: filters_applied.append(f"country_iso_code='{country_iso_code}'")
        if min_loan_amount_usd is not None: filters_applied.append(f"min_loan_amount_usd={min_loan_amount_usd}")
        if max_loan_amount_usd is not None: filters_applied.append(f"max_loan_amount_usd={max_loan_amount_usd}")
        if min_effective_pd_estimate is not None: filters_applied.append(f"min_effective_pd_estimate={min_effective_pd_estimate}")
        if max_effective_pd_estimate is not None: filters_applied.append(f"max_effective_pd_estimate={max_effective_pd_estimate}")
        if default_status is not None: filters_applied.append(f"default_status={default_status}")
        if hitl_overall_review_status is not None: filters_applied.append(f"hitl_overall_review_status='{hitl_overall_review_status.value}'")
        if min_effective_management_quality_score is not None: filters_applied.append(f"min_effective_mqs={min_effective_management_quality_score}")
        if max_effective_management_quality_score is not None: filters_applied.append(f"max_effective_mqs={max_effective_management_quality_score}")

        if filters_applied: log_message += " with filters: " + ", ".join(filters_applied)
        else: log_message += "..."
        logger.info(log_message)

        portfolio_risk_items: List[RiskItem] = []
        all_loans = self.kb_service.get_all_loans()
        if not all_loans:
            logger.warning("No loans found in Knowledge Base.")
            return []

        for loan in all_loans:
            company = self.kb_service.get_company_profile(loan.company_id)
            if not company:
                logger.warning(f"Company {loan.company_id} for loan {loan.loan_id} not found. Skipping.")
                continue

            model_pd: Optional[float] = None
            if self.pd_model.model is not None:
                pd_result = self.pd_model.predict_for_loan(loan.model_dump(), company.model_dump())
                if pd_result: _, model_pd = pd_result; model_pd = round(model_pd, 4) if model_pd is not None else None

            model_lgd: Optional[float] = None
            if self.lgd_model.model is not None:
                lgd_features = {'collateral_type': loan.collateral_type.value if loan.collateral_type else 'None', 'loan_amount_usd': loan.loan_amount, 'seniority_of_debt': str(loan.seniority_of_debt) if loan.seniority_of_debt else 'Unknown', 'economic_condition_indicator': loan.economic_condition_indicator if loan.economic_condition_indicator is not None else 0.5}
                raw_lgd = self.lgd_model.predict_lgd(lgd_features)
                if raw_lgd is not None: model_lgd = round(raw_lgd, 4)

            ead_val = loan.loan_amount
            model_el: Optional[float] = None
            if model_pd is not None and model_lgd is not None:
                model_el = round(model_pd * model_lgd * ead_val, 2)

            effective_pd = model_pd
            effective_lgd = model_lgd
            effective_mqs = company.management_quality_score
            has_pd_override, has_lgd_override, has_mqs_override = False, False, False

            loan_pd_anno = self.kb_service.get_latest_hitl_annotation_for_field(loan.loan_id, "loan", "pd_estimate")
            if loan_pd_anno and loan_pd_anno.new_value_numeric is not None:
                effective_pd = loan_pd_anno.new_value_numeric; has_pd_override = True

            loan_lgd_anno = self.kb_service.get_latest_hitl_annotation_for_field(loan.loan_id, "loan", "lgd_estimate")
            if loan_lgd_anno and loan_lgd_anno.new_value_numeric is not None:
                effective_lgd = loan_lgd_anno.new_value_numeric; has_lgd_override = True

            company_mqs_anno = self.kb_service.get_latest_hitl_annotation_for_field(company.company_id, "company", "management_quality_score")
            if company_mqs_anno and company_mqs_anno.new_value_numeric is not None:
                effective_mqs = int(company_mqs_anno.new_value_numeric); has_mqs_override = True

            effective_el: Optional[float] = None
            if effective_pd is not None and effective_lgd is not None:
                effective_el = round(effective_pd * effective_lgd * ead_val, 2)

            overall_hitl_status_val: Optional[HITLReviewStatus] = None
            last_hitl_timestamp_val: Optional[datetime.datetime] = None
            has_notes_val: bool = False
            latest_loan_overall_anno = self.kb_service.get_latest_hitl_annotation_for_field(loan.loan_id, "loan")
            latest_comp_overall_anno = self.kb_service.get_latest_hitl_annotation_for_field(company.company_id, "company")

            # Prioritize loan-specific annotation for overall status and timestamp

            if latest_loan_overall_anno:
                overall_hitl_status_val = latest_loan_overall_anno.hitl_review_status
                last_hitl_timestamp_val = latest_loan_overall_anno.annotation_timestamp
                if latest_loan_overall_anno.hitl_analyst_notes: has_notes_val = True
            elif latest_comp_overall_anno: # Fallback to company annotation

                overall_hitl_status_val = latest_comp_overall_anno.hitl_review_status
                last_hitl_timestamp_val = latest_comp_overall_anno.annotation_timestamp
                if latest_comp_overall_anno.hitl_analyst_notes: has_notes_val = True

            current_company_industry_sector_val = company.industry_sector.value if company.industry_sector else None
            if industry_sector and current_company_industry_sector_val != industry_sector: continue
            if country_iso_code and (not company.country_iso_code or company.country_iso_code.upper() != country_iso_code.upper()): continue
            if min_loan_amount_usd is not None and loan.loan_amount < min_loan_amount_usd: continue
            if max_loan_amount_usd is not None and loan.loan_amount > max_loan_amount_usd: continue
            if default_status is not None and loan.default_status != default_status: continue
            if min_effective_pd_estimate is not None and (effective_pd is None or effective_pd < min_effective_pd_estimate): continue
            if max_effective_pd_estimate is not None and (effective_pd is None or effective_pd > max_effective_pd_estimate): continue
            if hitl_overall_review_status is not None and (overall_hitl_status_val is None or overall_hitl_status_val != hitl_overall_review_status): continue
            if min_effective_management_quality_score is not None and (effective_mqs is None or effective_mqs < min_effective_management_quality_score): continue
            if max_effective_management_quality_score is not None and (effective_mqs is None or effective_mqs > max_effective_management_quality_score): continue

            kg_degree_centrality_val, kg_num_suppliers_val, kg_num_customers_val, kg_num_subsidiaries_val = None, None, None, None
            if self.kg_service and self.kg_service.graph.has_node(company.company_id):
                context_info = self.kg_service.get_company_contextual_info(company.company_id)
                if context_info:
                    raw_centrality = context_info.get('degree_centrality')
                    if raw_centrality is not None: kg_degree_centrality_val = round(raw_centrality, 4)
                    kg_num_suppliers_val = context_info.get('num_suppliers')
                    kg_num_customers_val = context_info.get('num_customers')
                    kg_num_subsidiaries_val = context_info.get('num_subsidiaries')

            try:
                risk_item_obj = RiskItem(
                    loan_id=loan.loan_id, company_id=company.company_id, company_name=company.company_name,
                    industry_sector=current_company_industry_sector_val, country_iso_code=company.country_iso_code,
                    founded_date=company.founded_date, loan_amount_usd=loan.loan_amount, currency=loan.currency.value,
                    collateral_type=loan.collateral_type.value if loan.collateral_type else None,
                    collateral_value_usd=loan.collateral_value_usd, is_defaulted=loan.default_status,
                    origination_date=loan.origination_date, maturity_date=loan.maturity_date,
                    interest_rate_percentage=loan.interest_rate_percentage, seniority_of_debt=loan.seniority_of_debt,
                    economic_condition_indicator=loan.economic_condition_indicator,
                    model_pd_estimate=model_pd, model_lgd_estimate=model_lgd, model_expected_loss_usd=model_el,
                    effective_pd_estimate=effective_pd, effective_lgd_estimate=effective_lgd, effective_expected_loss_usd=effective_el,
                    exposure_at_default_usd=ead_val,
                    original_management_quality_score=company.management_quality_score,
                    effective_management_quality_score=effective_mqs,
                    kg_degree_centrality=kg_degree_centrality_val, kg_num_suppliers=kg_num_suppliers_val,
                    kg_num_customers=kg_num_customers_val, kg_num_subsidiaries=kg_num_subsidiaries_val,
                    hitl_overall_review_status=overall_hitl_status_val,
                    hitl_last_annotation_timestamp=last_hitl_timestamp_val,
                    hitl_has_notes=has_notes_val, hitl_has_pd_override=has_pd_override,
                    hitl_has_lgd_override=has_lgd_override, hitl_has_mqs_override=has_mqs_override
                )
                portfolio_risk_items.append(risk_item_obj)
            except Exception as e:
                logger.error(f"Error creating RiskItem for loan {loan.loan_id}: {e}. Skipping.")
                logger.error(f"Data for failed RiskItem: loan_id={loan.loan_id}, company_id={company.company_id}, effective_pd={effective_pd}, effective_lgd={effective_lgd}")

        logger.info(f"Generated risk overview for {len(portfolio_risk_items)} loans after filtering.")
        return portfolio_risk_items

    def _generate_summary(self, portfolio_overview: List[RiskItem], group_by_field: str) -> Dict[str, Dict[str, Any]]:
        summary_data: Dict[str, Dict[str, Any]] = {}
        for item in portfolio_overview:
            group_key_val = getattr(item, group_by_field, None)
            group_key = str(group_key_val) if group_key_val is not None else "N/A"
            if group_key not in summary_data:
                summary_data[group_key] = {
                    "total_exposure": 0.0, "total_expected_loss": 0.0, "loan_count": 0,
                    "average_pd": [], "average_lgd": [], "defaulted_loan_count": 0,
                    "override_pd_count": 0, "override_lgd_count": 0, "override_mqs_count": 0
                }

            current_el = item.effective_expected_loss_usd if item.effective_expected_loss_usd is not None else 0.0
            current_ead = item.exposure_at_default_usd if item.exposure_at_default_usd is not None else 0.0
            summary_data[group_key]["total_exposure"] += current_ead
            summary_data[group_key]["total_expected_loss"] += current_el
            summary_data[group_key]["loan_count"] += 1
            if item.effective_pd_estimate is not None: summary_data[group_key]["average_pd"].append(item.effective_pd_estimate)
            if item.effective_lgd_estimate is not None: summary_data[group_key]["average_lgd"].append(item.effective_lgd_estimate)
            if item.is_defaulted: summary_data[group_key]["defaulted_loan_count"] +=1
            if item.hitl_has_pd_override: summary_data[group_key]["override_pd_count"] +=1
            if item.hitl_has_lgd_override: summary_data[group_key]["override_lgd_count"] +=1
            if item.hitl_has_mqs_override: summary_data[group_key]["override_mqs_count"] +=1

        for data in summary_data.values(): # Renamed to avoid conflict with outer 'data'

            data["average_pd"] = round(sum(data["average_pd"]) / len(data["average_pd"]), 4) if data["average_pd"] else 0.0
            data["average_lgd"] = round(sum(data["average_lgd"]) / len(data["average_lgd"]), 4) if data["average_lgd"] else 0.0
            data["total_expected_loss"] = round(data["total_expected_loss"], 2)
        return summary_data

    def get_risk_summary_by_sector(self, portfolio_overview: Optional[List[RiskItem]] = None, **filters) -> Dict[str, Dict[str, Any]]:
        if portfolio_overview is None:
            portfolio_overview = self.generate_portfolio_risk_overview(**filters)
        if not portfolio_overview:
            logger.warning("Portfolio overview is empty for sector summary. Applied filters might be too restrictive.")
            return {}
        sector_summary = self._generate_summary(portfolio_overview, group_by_field="industry_sector")
        logger.info(f"Generated risk summary for {len(sector_summary)} sectors.")
        return sector_summary

    def get_risk_summary_by_country(self, portfolio_overview: Optional[List[RiskItem]] = None, **filters) -> Dict[str, Dict[str, Any]]:
        if portfolio_overview is None:
            portfolio_overview = self.generate_portfolio_risk_overview(**filters)
        if not portfolio_overview:
            logger.warning("Portfolio overview is empty for country summary. Applied filters might be too restrictive.")
            return {}
        country_summary = self._generate_summary(portfolio_overview, group_by_field="country_iso_code")
        logger.info(f"Generated risk summary for {len(country_summary)} countries.")
        return country_summary

    def get_risk_summary_by_dimensions(self, dimensions: List[str], portfolio_overview: Optional[List[RiskItem]] = None, **filters) -> Dict[Any, Dict[str, Any]]:
        if not dimensions:
            logger.warning("No dimensions provided for summary.")
            return {}
        if portfolio_overview is None:
            portfolio_overview = self.generate_portfolio_risk_overview(**filters)
        if not portfolio_overview:
            logger.warning("Portfolio overview is empty for dimensional summary. Applied filters might be too restrictive.")
            return {}

        if len(dimensions) > 1:
            def composite_key_func(item):
                return tuple(str(getattr(item, dim, "N/A")) for dim in dimensions)

            summary_data_multi: Dict[tuple, Dict[str, Any]] = {} # Explicitly type for multi-dimension
            for item in portfolio_overview:
                group_key = composite_key_func(item)
                if group_key not in summary_data_multi:
                    summary_data_multi[group_key] = {

                        "total_exposure": 0.0, "total_expected_loss": 0.0, "loan_count": 0,
                        "average_pd": [], "average_lgd": [], "defaulted_loan_count": 0,
                        "override_pd_count": 0, "override_lgd_count": 0, "override_mqs_count": 0
                    }
                current_el = item.effective_expected_loss_usd if item.effective_expected_loss_usd is not None else 0.0
                current_ead = item.exposure_at_default_usd if item.exposure_at_default_usd is not None else 0.0
                summary_data_multi[group_key]["total_exposure"] += current_ead
                summary_data_multi[group_key]["total_expected_loss"] += current_el
                summary_data_multi[group_key]["loan_count"] += 1
                if item.effective_pd_estimate is not None: summary_data_multi[group_key]["average_pd"].append(item.effective_pd_estimate)
                if item.effective_lgd_estimate is not None: summary_data_multi[group_key]["average_lgd"].append(item.effective_lgd_estimate)
                if item.is_defaulted: summary_data_multi[group_key]["defaulted_loan_count"] +=1
                if item.hitl_has_pd_override: summary_data_multi[group_key]["override_pd_count"] +=1
                if item.hitl_has_lgd_override: summary_data_multi[group_key]["override_lgd_count"] +=1
                if item.hitl_has_mqs_override: summary_data_multi[group_key]["override_mqs_count"] +=1

            for data_val in summary_data_multi.values():
                data_val["average_pd"] = round(sum(data_val["average_pd"]) / len(data_val["average_pd"]), 4) if data_val["average_pd"] else 0.0
                data_val["average_lgd"] = round(sum(data_val["average_lgd"]) / len(data_val["average_lgd"]), 4) if data_val["average_lgd"] else 0.0
                data_val["total_expected_loss"] = round(data_val["total_expected_loss"], 2)
            logger.info(f"Generated multi-dimensional risk summary by {dimensions} for {len(summary_data_multi)} groups.")
            return summary_data_multi # type: ignore
        else:
            group_by_field = dimensions[0]
            # Basic check if field exists using RiskItem.model_fields
            if group_by_field not in RiskItem.model_fields:

                 logger.error(f"Invalid dimension for summary: {group_by_field}")
                 return {}
            summary = self._generate_summary(portfolio_overview, group_by_field=group_by_field)
            logger.info(f"Generated risk summary by {group_by_field} for {len(summary)} groups.")
            return summary


if __name__ == "__main__":
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- Testing RiskMapService (Extended HITL & Filtering) ---")
    kb = KnowledgeBaseService()

    pd_m = PDModel()
    lgd_m = LGDModel()

    if not pd_m.load_model():
        logger.warning("PD model file not found. Training PD model for RiskMapService test...")
        if kb.get_all_loans() and kb.get_all_companies(): pd_m.train(kb_service=kb)
        else: logger.error("Cannot train PD model: No data in KB.")
    if not lgd_m.load_model():
        logger.warning("LGD model file not found. Training LGD model for RiskMapService test...")
        if kb.get_all_loans(): lgd_m.train(kb_service=kb)
        else: logger.error("Cannot train LGD model: No data in KB.")

    if pd_m.model and lgd_m.model:
        kg_service = KnowledgeGraphService(kb_service=kb)
        risk_map_service = RiskMapService(kb_service=kb, pd_model=pd_m, lgd_model=lgd_m, kg_service=kg_service)

        logger.info("\n--- Full Portfolio Overview (First item if available) ---")
        portfolio_all = risk_map_service.generate_portfolio_risk_overview()
        logger.info(f"Total items in full overview: {len(portfolio_all)}")
        if portfolio_all:
            logger.info(f"First Item: {portfolio_all[0].model_dump_json(indent=2, exclude_none=True)}")


        logger.info("\n--- Filtered: Technology Sector ---")
        portfolio_tech = risk_map_service.generate_portfolio_risk_overview(industry_sector="Technology")
        logger.info(f"Tech items: {len(portfolio_tech)}. First if available: {portfolio_tech[0].loan_id if portfolio_tech else 'None'}")

        logger.info("\n--- Filtered: Effective PD > 0.07 ---")
        portfolio_pd_filtered = risk_map_service.generate_portfolio_risk_overview(min_effective_pd_estimate=0.07)
        logger.info(f"Items with Effective PD > 0.07: {len(portfolio_pd_filtered)}")
        for item in portfolio_pd_filtered: # Limit output for brevity
            if item.loan_id == "LOAN7002": # LOAN7002 has PD override to 0.085 in sample_hitl_annotations
                 logger.info(f"  Loan: {item.loan_id}, Company: {item.company_id}, ModelPD: {item.model_pd_estimate}, EffectivePD: {item.effective_pd_estimate}, Override: {item.hitl_has_pd_override}")

        logger.info("\n--- Filtered: HITL Status - FLAGGED_MODEL_DISAGREEMENT ---")

        portfolio_hitl_flagged = risk_map_service.generate_portfolio_risk_overview(
            hitl_overall_review_status=HITLReviewStatus.FLAGGED_MODEL_DISAGREEMENT
        )
        logger.info(f"Items with HITL Status FLAGGED_MODEL_DISAGREEMENT: {len(portfolio_hitl_flagged)}")
        if portfolio_hitl_flagged:
             logger.info(f"  First flagged item loan_id: {portfolio_hitl_flagged[0].loan_id}, status: {portfolio_hitl_flagged[0].hitl_overall_review_status.value if portfolio_hitl_flagged[0].hitl_overall_review_status else 'N/A'}")

        logger.info("\n--- Filtered: Effective MQS < 7 ---")
        portfolio_mqs_filtered = risk_map_service.generate_portfolio_risk_overview(max_effective_management_quality_score=6)
        logger.info(f"Items with Effective MQS < 7: {len(portfolio_mqs_filtered)}")
        # for item in portfolio_mqs_filtered:
        #      logger.info(f"  Loan: {item.loan_id}, Company: {item.company_id}, EffectiveMQS: {item.effective_management_quality_score}, Override: {item.hitl_has_mqs_override}")

        logger.info("\n--- Summary by Sector (on full portfolio) ---")
        sector_summary = risk_map_service.get_risk_summary_by_sector(portfolio_overview=portfolio_all) # Can also do get_risk_summary_by_sector() to use all filters
        logger.info(json.dumps(sector_summary, indent=2, default=str))

        logger.info("\n--- Summary by Country (filtered for Technology) ---")
        country_summary_tech = risk_map_service.get_risk_summary_by_country(industry_sector="Technology") # Pass filter here
        logger.info(json.dumps(country_summary_tech, indent=2, default=str))

        logger.info("\n--- Summary by 'is_defaulted' (generic dimensions on full portfolio) ---")
        default_summary = risk_map_service.get_risk_summary_by_dimensions(dimensions=["is_defaulted"], portfolio_overview=portfolio_all)
        logger.info(json.dumps(default_summary, indent=2, default=str))

        logger.info("\n--- Summary by 'industry_sector' and 'country_iso_code' (generic multi-dimensions) ---")
        multi_dim_summary = risk_map_service.get_risk_summary_by_dimensions(
            dimensions=["industry_sector", "country_iso_code"],
            portfolio_overview=portfolio_all
        )
        logger.info(f"Multi-dim summary groups: {len(multi_dim_summary)}")
        # Example of accessing a multi-dim key if needed for print:
        # for k, v in multi_dim_summary.items():
        #     logger.info(f"Group {str(k)}: {v['loan_count']} loans")



    else:
        logger.error("RiskMapService could not be initialized due to missing models. Tests aborted.")
