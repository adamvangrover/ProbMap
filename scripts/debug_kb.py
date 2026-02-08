import sys
import os
import logging
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data_management.knowledge_base import KnowledgeBaseService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_kb():
    kb = KnowledgeBaseService()
    companies = kb.get_all_companies()
    logger.info(f"Loaded {len(companies)} companies.")
    if companies:
        logger.info(f"First 5 IDs: {[c.company_id for c in companies[:5]]}")
        logger.info(f"First company data: {companies[0].model_dump()}")

    loans = kb.get_all_loans()
    logger.info(f"Loaded {len(loans)} loans.")

    # Check specific join
    if loans and companies:
        l = loans[0]
        c = kb.get_company_profile(l.company_id)
        if c:
            logger.info(f"Loan {l.loan_id} links to Company {c.company_id} successfully.")
        else:
            logger.warning(f"Loan {l.loan_id} links to {l.company_id} BUT NOT FOUND in KB.")

if __name__ == "__main__":
    debug_kb()
