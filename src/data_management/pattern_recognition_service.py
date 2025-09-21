import logging
from typing import List, Dict, Any

from src.data_management.knowledge_graph import KnowledgeGraphService, RelationshipType

logger = logging.getLogger(__name__)

class PatternRecognitionService:
    """
    A service to find specific, predefined patterns in the knowledge graph.
    These patterns can represent complex risk scenarios.
    """

    def __init__(self, kg_service: KnowledgeGraphService):
        self.kg_service = kg_service
        if self.kg_service.graph is None or self.kg_service.graph.number_of_nodes() == 0:
            logger.warning("PatternRecognitionService initialized with an empty or unpopulated KnowledgeGraphService.")

    def find_companies_with_high_risk_suppliers(self, default_days_threshold: int = 90) -> List[Dict[str, Any]]:
        """
        Finds companies that are connected to suppliers who have defaulted on loans.
        A supplier is considered "high-risk" if they have a recorded default event.

        Args:
            default_days_threshold (int): This argument is not used in the current implementation,
                                           but could be used in a more advanced version to check
                                           for recent defaults.

        Returns:
            A list of dictionaries, where each dictionary represents a company
            and contains the company's ID, name, and a list of its high-risk suppliers.
        """
        logger.info("Starting pattern search for companies with high-risk suppliers.")
        companies_with_risky_suppliers = []
        
        if self.kg_service.graph is None:
            logger.error("Knowledge graph is not available.")
            return companies_with_risky_suppliers

        # Iterate through all nodes in the graph to find companies
        for node_id, node_data in self.kg_service.graph.nodes(data=True):
            if node_data.get("type") == "CorporateEntity":
                company_id = node_id
                company_name = node_data.get("company_name", "N/A")
                
                risky_suppliers_for_company = []

                # Get all suppliers for the current company
                suppliers = self.kg_service.get_neighbors(company_id, relationship_type=RelationshipType.HAS_SUPPLIER)

                for supplier_id in suppliers:
                    # For each supplier, check if they have defaulted on any loan
                    # A default is marked by a HAS_DEFAULTED_ON_LOAN edge from the supplier to a loan
                    supplier_defaulted_loans = self.kg_service.get_neighbors(supplier_id, relationship_type=RelationshipType.HAS_DEFAULTED_ON_LOAN)
                    
                    if supplier_defaulted_loans:
                        supplier_info = self.kg_service.get_node_attributes(supplier_id)
                        supplier_name = supplier_info.get("company_name", supplier_id) if supplier_info else supplier_id
                        
                        risky_suppliers_for_company.append({
                            "supplier_id": supplier_id,
                            "supplier_name": supplier_name,
                            "defaulted_loan_ids": supplier_defaulted_loans
                        })

                if risky_suppliers_for_company:
                    companies_with_risky_suppliers.append({
                        "company_id": company_id,
                        "company_name": company_name,
                        "high_risk_suppliers": risky_suppliers_for_company
                    })
        
        logger.info(f"Found {len(companies_with_risky_suppliers)} companies with high-risk suppliers.")
        return companies_with_risky_suppliers
