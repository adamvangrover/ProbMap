import networkx as nx
import logging
from typing import List, Tuple, Dict, Any, Optional

from src.data_management.ontology import CorporateEntity, LoanAgreement # For type hinting
from src.data_management.knowledge_base import KnowledgeBaseService # To populate graph

logger = logging.getLogger(__name__)

class RelationshipType(str):
    HAS_LOAN = "HAS_LOAN"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    SUPPLIER_TO = "SUPPLIER_TO"
    CUSTOMER_OF = "CUSTOMER_OF"
    LOCATED_IN_SECTOR = "LOCATED_IN_SECTOR"
    LOCATED_IN_COUNTRY = "LOCATED_IN_COUNTRY"
    HAS_DEFAULTED_ON_LOAN = "HAS_DEFAULTED_ON_LOAN"

class KnowledgeGraphService:
    """
    Manages the creation and querying of a knowledge graph representing
    entities and their relationships.
    For PoC, uses NetworkX for an in-memory graph.
    """
    def __init__(self, kb_service: Optional[KnowledgeBaseService] = None):
        self.graph = nx.MultiDiGraph() # MultiDiGraph allows multiple edges between nodes and directed edges
        self.kb_service = kb_service
        if self.kb_service:
            self._populate_graph_from_kb()

    def _populate_graph_from_kb(self):
        """Populates the graph with data from the KnowledgeBaseService."""
        if not self.kb_service:
            logger.warning("KnowledgeBaseService not provided, cannot populate graph.")
            return

        logger.info("Populating Knowledge Graph from Knowledge Base...")

        # Add company nodes
        all_companies = self.kb_service.get_all_companies()
        for company in all_companies:
            self.add_node(company.company_id, node_type="CorporateEntity", **company.model_dump())
            # Add relationships for sector and country
            self.add_node(company.industry_sector.value, node_type="IndustrySector")
            self.add_edge(company.company_id, company.industry_sector.value, RelationshipType.LOCATED_IN_SECTOR)

            self.add_node(company.country_iso_code, node_type="Country")
            self.add_edge(company.company_id, company.country_iso_code, RelationshipType.LOCATED_IN_COUNTRY)

        # Add loan nodes and relationships
        all_loans = self.kb_service.get_all_loans()
        for loan in all_loans:
            self.add_node(loan.loan_id, node_type="LoanAgreement", **loan.model_dump())
            # Link company to loan
            if self.graph.has_node(loan.company_id):
                self.add_edge(loan.company_id, loan.loan_id, RelationshipType.HAS_LOAN)
                if loan.default_status:
                    self.add_edge(loan.company_id, loan.loan_id, RelationshipType.HAS_DEFAULTED_ON_LOAN, defaulted_on=loan.default_date)
            else:
                logger.warning(f"Company ID {loan.company_id} for loan {loan.loan_id} not found in graph nodes.")

        logger.info(f"Graph populated. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")


    def add_node(self, node_id: str, node_type: str, **attributes: Any):
        """Adds a node to the graph if it doesn't already exist."""
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type=node_type, **attributes)
            # logger.debug(f"Added node: {node_id} (Type: {node_type})")
        else:
            # Optionally update attributes if node exists
            # logger.debug(f"Node {node_id} already exists. Attributes: {attributes}")
            self.graph.nodes[node_id].update(attributes)


    def add_edge(self, source_node_id: str, target_node_id: str, relationship_type: str, **attributes: Any):
        """Adds a directed edge between two nodes if they exist."""
        if self.graph.has_node(source_node_id) and self.graph.has_node(target_node_id):
            self.graph.add_edge(source_node_id, target_node_id, key=relationship_type, type=relationship_type, **attributes)
            # logger.debug(f"Added edge: {source_node_id} -[{relationship_type}]-> {target_node_id}")
        else:
            if not self.graph.has_node(source_node_id):
                logger.warning(f"Source node {source_node_id} not found. Cannot add edge.")
            if not self.graph.has_node(target_node_id):
                logger.warning(f"Target node {target_node_id} not found. Cannot add edge.")

    def get_node_attributes(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves attributes of a specific node."""
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id]
        logger.warning(f"Node {node_id} not found.")
        return None

    def get_neighbors(self, node_id: str, relationship_type: Optional[str] = None) -> List[str]:
        """
        Retrieves neighbors of a node, optionally filtered by relationship type.
        Returns a list of neighbor node IDs.
        """
        if not self.graph.has_node(node_id):
            logger.warning(f"Node {node_id} not found for neighbor search.")
            return []

        neighbors = []
        # For outgoing edges (successors)
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relationship_type is None or data.get('type') == relationship_type:
                neighbors.append(target)
        # For incoming edges (predecessors) - uncomment if needed
        # for source, _, data in self.graph.in_edges(node_id, data=True):
        #     if relationship_type is None or data.get('type') == relationship_type:
        #         if source not in neighbors: # Avoid duplicates if also successor
        #             neighbors.append(source)
        return list(set(neighbors)) # Ensure uniqueness

    def find_paths(self, source_node_id: str, target_node_id: str, cutoff: Optional[int] = None) -> List[List[str]]:
        """Finds all simple paths between two nodes."""
        if not self.graph.has_node(source_node_id) or not self.graph.has_node(target_node_id):
            logger.warning(f"Source or target node not found for path search.")
            return []
        try:
            return list(nx.all_simple_paths(self.graph, source=source_node_id, target=target_node_id, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []

    def get_related_entities_by_intermediary(self, start_node_id: str, relation1: str, relation2: str) -> List[str]:
        """
        Finds entities related to start_node_id through an intermediary node.
        E.g., Find other companies (target) that have loans (relation2) with the same bank (intermediary)
        that the start_node_id (company) also has a loan (relation1) with.
        This is a conceptual example; actual relations depend on graph structure.
        """
        related = []
        if not self.graph.has_node(start_node_id):
            return related

        for intermediary in self.get_neighbors(start_node_id, relationship_type=relation1):
            for target_entity in self.get_neighbors(intermediary, relationship_type=relation2):
                if target_entity != start_node_id and target_entity not in related:
                    related.append(target_entity)
        return related

    def get_company_default_history(self, company_id: str) -> List[Dict[str, Any]]:
        default_events = []
        if not self.graph.has_node(company_id):
            logger.warning(f"Company node {company_id} not found.")
            return default_events

        for _, loan_node_id, data in self.graph.out_edges(company_id, data=True):
            if data.get('type') == RelationshipType.HAS_DEFAULTED_ON_LOAN:
                loan_attributes = self.get_node_attributes(loan_node_id)
                if loan_attributes:
                    default_event = {
                        "loan_id": loan_node_id,
                        "default_date": data.get("defaulted_on", "N/A"), # Assuming this attribute was added on edge creation
                        "loan_amount": loan_attributes.get("loan_amount"),
                        "currency": loan_attributes.get("currency")
                    }
                    default_events.append(default_event)
        return default_events


if __name__ == "__main__":
    # This requires src.core.logging_config and src.data_management.knowledge_base to be importable
    # Run from project root: python -m src.data_management.knowledge_graph

    # from src.core.logging_config import setup_logging # Explicitly call if not auto-setup
    # setup_logging()

    logger.info("--- Testing KnowledgeGraphService ---")
    kb = KnowledgeBaseService() # Load data into KB
    kg_service = KnowledgeGraphService(kb_service=kb) # Populate KG from KB

    # Test node and edge counts
    logger.info(f"Graph nodes: {kg_service.graph.number_of_nodes()}, Edges: {kg_service.graph.number_of_edges()}")

    # Test get_node_attributes
    test_company_id = "COMP001"
    node_attrs = kg_service.get_node_attributes(test_company_id)
    if node_attrs:
        logger.info(f"Attributes for {test_company_id}: Name: {node_attrs.get('company_name')}, Type: {node_attrs.get('type')}")

    test_loan_id = "LOAN7001"
    loan_attrs = kg_service.get_node_attributes(test_loan_id)
    if loan_attrs:
        logger.info(f"Attributes for {test_loan_id}: Amount: {loan_attrs.get('loan_amount_usd')}, Type: {loan_attrs.get('type')}")

    # Test get_neighbors
    comp1_loans = kg_service.get_neighbors(test_company_id, relationship_type=RelationshipType.HAS_LOAN)
    logger.info(f"Loans for {test_company_id}: {comp1_loans}")

    comp1_sector = kg_service.get_neighbors(test_company_id, relationship_type=RelationshipType.LOCATED_IN_SECTOR)
    logger.info(f"Sector for {test_company_id}: {comp1_sector}")


    # Test find_paths (simple direct path)
    # Example: Path from a company to its loan
    if comp1_loans:
        paths = kg_service.find_paths(test_company_id, comp1_loans[0])
        logger.info(f"Paths from {test_company_id} to {comp1_loans[0]}: {paths}")

    # Test a more complex query if data allows (e.g., companies in the same sector)
    target_sector = "Technology" # Assuming 'Technology' is a node from ontology.py
    if kg_service.graph.has_node(target_sector):
        companies_in_sector = []
        for source_node, _, data in kg_service.graph.in_edges(target_sector, data=True):
             if data.get('type') == RelationshipType.LOCATED_IN_SECTOR:
                node_data = kg_service.get_node_attributes(source_node)
                if node_data and node_data.get('type') == "CorporateEntity":
                    companies_in_sector.append(source_node)
        logger.info(f"Companies in '{target_sector}': {companies_in_sector}")
    else:
        logger.warning(f"Sector node '{target_sector}' not found in graph.")

    # Test default history
    defaulting_company_id = "COMP004" # This company has a defaulted loan in sample data
    default_history = kg_service.get_company_default_history(defaulting_company_id)
    logger.info(f"Default history for {defaulting_company_id}: {default_history}")

    non_defaulting_company_id = "COMP001"
    default_history_non = kg_service.get_company_default_history(non_defaulting_company_id)
    logger.info(f"Default history for {non_defaulting_company_id}: {default_history_non}")


    # Example of adding a new relationship not derived from KB
    # kg_service.add_node("BankA", node_type="FinancialInstitution", name="Global Bank Corp")
    # kg_service.add_node("BankB", node_type="FinancialInstitution", name="Local Credit Union")
    # kg_service.add_edge("COMP001", "BankA", relationship_type="HAS_ACCOUNT_WITH")
    # kg_service.add_edge("COMP002", "BankA", relationship_type="HAS_ACCOUNT_WITH")
    # kg_service.add_edge("COMP003", "BankB", relationship_type="HAS_ACCOUNT_WITH")

    # Conceptual: Find companies that share the same bank (BankA) as COMP001
    # This requires the above commented out additions to be made first
    # shared_bank_companies = kg_service.get_related_entities_by_intermediary(
    #     start_node_id="COMP001",
    #     relation1="HAS_ACCOUNT_WITH", # company -> bank
    #     relation2="HAS_ACCOUNT_WITH"  # bank -> company (reversed in typical neighbor search)
    # )
    # logger.info(f"Companies sharing a bank with COMP001 (via BankA - conceptual): {shared_bank_companies}")
