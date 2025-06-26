import networkx as nx
import logging
from typing import List, Tuple, Dict, Any, Optional

from src.data_management.ontology import CorporateEntity, LoanAgreement # For type hinting
from src.data_management.knowledge_base import KnowledgeBaseService # To populate graph

logger = logging.getLogger(__name__)

class RelationshipType(str):
    HAS_LOAN = "HAS_LOAN"
    SUBSIDIARY_OF = "SUBSIDIARY_OF" # Child -> Parent
    HAS_SUBSIDIARY = "HAS_SUBSIDIARY" # Parent -> Child

    SUPPLIER_TO = "SUPPLIER_TO"     # Supplier -> Company
    HAS_SUPPLIER = "HAS_SUPPLIER"   # Company -> Supplier

    CUSTOMER_OF = "CUSTOMER_OF"     # Customer -> Company
    HAS_CUSTOMER = "HAS_CUSTOMER"   # Company -> Customer

    LOCATED_IN_SECTOR = "LOCATED_IN_SECTOR"
    LOCATED_IN_COUNTRY = "LOCATED_IN_COUNTRY"

    HAS_DEFAULTED_ON_LOAN = "HAS_DEFAULTED_ON_LOAN" # Company -> Loan (special case, could be on Loan node itself)

    # New relationship types
    HAS_FINANCIAL_STATEMENT = "HAS_FINANCIAL_STATEMENT" # Company -> FinancialStatement
    HAS_DEFAULT_EVENT = "HAS_DEFAULT_EVENT"             # Loan -> DefaultEvent

    GUARANTEES = "GUARANTEES"                           # Guarantor (Company) -> Loan
    HAS_GUARANTOR = "HAS_GUARANTOR"                     # Loan -> Guarantor (Company)


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
        company_ids_in_kb = {c.company_id for c in all_companies}

        for company in all_companies:
            company_attributes = company.model_dump()
            # Ensure list fields are not None for the purpose of iteration, if Pydantic model_dump might make them None
            company_attributes['subsidiaries'] = company_attributes.get('subsidiaries') or []
            company_attributes['suppliers'] = company_attributes.get('suppliers') or []
            company_attributes['customers'] = company_attributes.get('customers') or []

            self.add_node(company.company_id, node_type="CorporateEntity", **company_attributes)

            # Add relationships for sector and country
            if company.industry_sector: # Check if not None
                self.add_node(company.industry_sector.value, node_type="IndustrySector")
                self.add_edge(company.company_id, company.industry_sector.value, RelationshipType.LOCATED_IN_SECTOR)
            if company.country_iso_code: # Check if not None
                self.add_node(company.country_iso_code, node_type="Country")
                self.add_edge(company.company_id, company.country_iso_code, RelationshipType.LOCATED_IN_COUNTRY)

            # Process subsidiaries
            for subsidiary_id in company_attributes['subsidiaries']:
                if subsidiary_id in company_ids_in_kb: # Ensure subsidiary is a known company
                     self.add_node(subsidiary_id, node_type="CorporateEntity")
                     self.add_edge(company.company_id, subsidiary_id, RelationshipType.HAS_SUBSIDIARY)
                     self.add_edge(subsidiary_id, company.company_id, RelationshipType.SUBSIDIARY_OF)
                else:
                    logger.warning(f"Subsidiary ID {subsidiary_id} for company {company.company_id} not found in KB companies. Node created, but may lack attributes.")
                    self.add_node(subsidiary_id, node_type="CorporateEntity", name=f"Placeholder {subsidiary_id}", is_placeholder=True)


            # Process suppliers
            for supplier_id in company_attributes['suppliers']:
                if supplier_id not in company_ids_in_kb: # If supplier is not a known full company entity
                    self.add_node(supplier_id, node_type="CorporateEntity", name=f"Placeholder {supplier_id}", is_placeholder=True)
                else: # Supplier is a known company, ensure it's added (attributes might be updated later)
                    self.add_node(supplier_id, node_type="CorporateEntity")
                self.add_edge(company.company_id, supplier_id, RelationshipType.HAS_SUPPLIER)
                self.add_edge(supplier_id, company.company_id, RelationshipType.SUPPLIER_TO)

            # Process customers
            for customer_id in company_attributes['customers']:
                if customer_id not in company_ids_in_kb: # If customer is not a known full company entity
                    self.add_node(customer_id, node_type="CorporateEntity", name=f"Placeholder {customer_id}", is_placeholder=True)
                else: # Customer is a known company
                    self.add_node(customer_id, node_type="CorporateEntity")
                self.add_edge(company.company_id, customer_id, RelationshipType.HAS_CUSTOMER)
                self.add_edge(customer_id, company.company_id, RelationshipType.CUSTOMER_OF)

            # Process financial statements for the company
            statements = self.kb_service.get_financial_statements_for_company(company.company_id)
            for statement in statements:
                self.add_node(statement.statement_id, node_type="FinancialStatement", **statement.model_dump())
                self.add_edge(company.company_id, statement.statement_id, RelationshipType.HAS_FINANCIAL_STATEMENT)

        # Add loan nodes and relationships
        all_loans = self.kb_service.get_all_loans()
        for loan in all_loans:
            loan_attributes = loan.model_dump()
            loan_attributes['guarantors'] = loan_attributes.get('guarantors') or []

            self.add_node(loan.loan_id, node_type="LoanAgreement", **loan_attributes)

            # Link company to loan
            if self.graph.has_node(loan.company_id):
                self.add_edge(loan.company_id, loan.loan_id, RelationshipType.HAS_LOAN)
                if loan.default_status: # The HAS_DEFAULTED_ON_LOAN is company -> loan
                    # The default date is on the DefaultEvent, not directly on the loan edge here
                    self.add_edge(loan.company_id, loan.loan_id, RelationshipType.HAS_DEFAULTED_ON_LOAN)
            else:
                logger.warning(f"Company ID {loan.company_id} for loan {loan.loan_id} not found in graph nodes.")

            # Process guarantors for the loan
            for guarantor_id in loan_attributes['guarantors']:
                if self.graph.has_node(guarantor_id): # Guarantor must be an existing company node
                    self.add_edge(guarantor_id, loan.loan_id, RelationshipType.GUARANTEES)
                    self.add_edge(loan.loan_id, guarantor_id, RelationshipType.HAS_GUARANTOR)
                else:
                    logger.warning(f"Guarantor ID {guarantor_id} for loan {loan.loan_id} not found in graph. Edge not added.")

            # Process default events for the loan
            events = self.kb_service.get_default_events_for_loan(loan.loan_id)
            for event in events:
                self.add_node(event.event_id, node_type="DefaultEvent", **event.model_dump())
                self.add_edge(loan.loan_id, event.event_id, RelationshipType.HAS_DEFAULT_EVENT)
                # If the loan is defaulted, also ensure the company -> loan defaulted edge exists.
                # This might be redundant if loan.default_status already handled it, but good for consistency.
                if self.graph.has_node(loan.company_id):
                     self.add_edge(loan.company_id, loan.loan_id, RelationshipType.HAS_DEFAULTED_ON_LOAN, event_date=event.default_date)


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

    def find_common_guarantors(self, loan_ids: List[str]) -> Dict[str, List[str]]:
        """
        Finds common guarantors for a given list of loan IDs.
        Returns a dictionary where keys are guarantor company_ids and values are lists of loan_ids
        from the input list that they guarantee.
        """
        guarantor_map: Dict[str, List[str]] = {} # Maps guarantor_id to list of loans they guarantee
        if not self.graph:
            return guarantor_map

        for loan_id in loan_ids:
            if not self.graph.has_node(loan_id):
                logger.warning(f"Loan ID {loan_id} not found in graph for common guarantor search.")
                continue
            # Guarantors are connected to loans via an incoming GUARANTEES edge from the guarantor
            # or an outgoing HAS_GUARANTOR edge to the guarantor. We used both.
            # Let's use the HAS_GUARANTOR edge from Loan to Guarantor Entity
            for _, guarantor_node_id, data in self.graph.out_edges(loan_id, data=True):
                if data.get('type') == RelationshipType.HAS_GUARANTOR:
                    if guarantor_node_id not in guarantor_map:
                        guarantor_map[guarantor_node_id] = []
                    if loan_id not in guarantor_map[guarantor_node_id]: # ensure loan_id is added once per guarantor
                        guarantor_map[guarantor_node_id].append(loan_id)

        # Filter for guarantors common to more than one loan in the input list
        common_guarantors = {
            gid: loans for gid, loans in guarantor_map.items() if len(loans) > 1 and any(l_id in loan_ids for l_id in loans)
        }
        # Ensure that the loans listed for each common guarantor are only those from the input loan_ids list
        for gid in list(common_guarantors.keys()): # Iterate over a copy of keys for safe modification
            common_guarantors[gid] = [l_id for l_id in common_guarantors[gid] if l_id in loan_ids]
            if not common_guarantors[gid] or len(common_guarantors[gid]) <=1 : # clean up if no longer common to input loans after filtering
                del common_guarantors[gid]

        return common_guarantors

    def get_entity_centrality(self, entity_id: str, algorithm: str = 'degree') -> float:
        """Calculates centrality for a given entity_id."""
        if not self.graph.has_node(entity_id):
            logger.warning(f"Entity {entity_id} not found in graph. Cannot calculate centrality.")
            return 0.0

        try:
            if algorithm == 'degree':
                # Ensure graph is not empty for degree_centrality
                if not self.graph: return 0.0
                centrality = nx.degree_centrality(self.graph)
            elif algorithm == 'betweenness':
                # Ensure graph has more than 2 nodes for betweenness_centrality
                if self.graph.number_of_nodes() <= 2: return 0.0
                centrality = nx.betweenness_centrality(self.graph)
            elif algorithm == 'closeness':
                # Ensure graph is connected for closeness_centrality or handle disconnected components
                # For simplicity, we calculate for the component containing the node if graph is not connected
                # However, NetworkX closeness_centrality handles disconnected graphs by calculating for each component.
                if not nx.is_connected(self.graph.to_undirected()): # Check overall connectivity
                    # Get the component containing the node
                    component_nodes = nx.node_connected_component(self.graph.to_undirected(), entity_id)
                    subgraph = self.graph.subgraph(component_nodes)
                    if not subgraph: return 0.0
                    centrality = nx.closeness_centrality(subgraph) # Calculate on the component
                else:
                    centrality = nx.closeness_centrality(self.graph)
            else:
                logger.warning(f"Unknown centrality algorithm: {algorithm}. Supported: degree, betweenness, closeness.")
                return 0.0

            return centrality.get(entity_id, 0.0) # Get score for specific entity, default 0.0 if not in result
        except Exception as e: # Catch any NetworkX errors or other issues
            logger.error(f"Error calculating {algorithm} centrality for {entity_id}: {e}")
            return 0.0

    def find_paths_between_entities(self, source_id: str, target_id: str,
                                    relationship_types: Optional[List[str]] = None,
                                    cutoff: int = 5) -> List[List[str]]:
        """
        Finds simple paths between two entities.
        If relationship_types is provided, paths are filtered to include only edges of those types.
        (Simplified PoC: all edges in a path must match one of the provided types).
        """
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            logger.warning(f"Source ({source_id}) or Target ({target_id}) node not found for path search.")
            return []

        try:
            all_paths = list(nx.all_simple_paths(self.graph, source=source_id, target=target_id, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
             logger.warning(f"Node {source_id} or {target_id} not found during path search with nx.all_simple_paths.")
             return []


        if not relationship_types:
            return all_paths

        filtered_paths = []
        for path in all_paths:
            is_valid_path = True
            if len(path) < 2: # Path must have at least two nodes to have an edge
                if not relationship_types: # if no filter, single node path is valid if it's source==target
                    filtered_paths.append(path)
                continue

            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Check all edges between u and v in the MultiDiGraph
                edge_data_list = self.graph.get_edge_data(u, v)
                if not edge_data_list: # Should not happen in a path from all_simple_paths
                    is_valid_path = False
                    break

                # For MultiDiGraph, get_edge_data returns a dict like {key: data_dict, ...}
                # We need to check if *any* edge between u and v (that could form this path segment)
                # matches the required relationship types.
                # all_simple_paths doesn't specify *which* edge if multiple exist.
                # This simplification assumes any edge between u,v matching types is fine for this path segment.

                found_matching_edge_for_segment = False
                for edge_key, data_dict in edge_data_list.items():
                    if data_dict.get('type') in relationship_types:
                        found_matching_edge_for_segment = True
                        break
                if not found_matching_edge_for_segment:
                    is_valid_path = False
                    break

            if is_valid_path:
                filtered_paths.append(path)

        return filtered_paths

    def get_company_contextual_info(self, company_id: str) -> Dict[str, Any]:
        """Retrieves basic KG-derived contextual information for a given company_id."""
        context_info: Dict[str, Any] = {
            "company_id": company_id,
            "node_exists": False,
            "num_loans": 0,
            "num_defaulted_loans": 0,
            "num_financial_statements": 0,
            "num_subsidiaries": 0,
            "num_suppliers": 0,
            "num_customers": 0,
            "degree_centrality": 0.0,
        }

        if not self.graph.has_node(company_id):
            logger.warning(f"Company node {company_id} not found in graph for contextual info.")
            return context_info

        context_info["node_exists"] = True
        node_attributes = self.get_node_attributes(company_id)
        if node_attributes and node_attributes.get('type') != "CorporateEntity":
            logger.warning(f"Node {company_id} is not a CorporateEntity. Contextual info might be limited/irrelevant.")
            # Still proceed to calculate edge-based counts as they might be valid depending on graph modeling

        # Count outgoing edges for various relationships
        num_loans = 0
        num_defaulted_loans = 0
        num_financial_statements = 0
        num_subsidiaries = 0
        num_suppliers = 0
        num_customers = 0

        if self.graph.has_node(company_id): # Redundant check, but good for safety
            for _, target, data in self.graph.out_edges(company_id, data=True):
                edge_type = data.get('type')
                if edge_type == RelationshipType.HAS_LOAN:
                    num_loans += 1
                elif edge_type == RelationshipType.HAS_DEFAULTED_ON_LOAN: # Company -> Loan
                    num_defaulted_loans +=1
                elif edge_type == RelationshipType.HAS_FINANCIAL_STATEMENT:
                    num_financial_statements += 1
                elif edge_type == RelationshipType.HAS_SUBSIDIARY:
                    num_subsidiaries += 1
                elif edge_type == RelationshipType.HAS_SUPPLIER:
                    num_suppliers += 1
                elif edge_type == RelationshipType.HAS_CUSTOMER:
                    num_customers += 1

        context_info["num_loans"] = num_loans
        context_info["num_defaulted_loans"] = num_defaulted_loans
        context_info["num_financial_statements"] = num_financial_statements
        context_info["num_subsidiaries"] = num_subsidiaries
        context_info["num_suppliers"] = num_suppliers
        context_info["num_customers"] = num_customers

        try:
            context_info["degree_centrality"] = self.get_entity_centrality(company_id, algorithm='degree')
        except Exception as e:
            logger.error(f"Error calculating degree centrality for {company_id} in contextual info: {e}")
            context_info["degree_centrality"] = 0.0 # Default on error

        return context_info


if __name__ == "__main__":
    # This requires src.core.logging_config and src.data_management.knowledge_base to be importable
    # Run from project root: python -m src.data_management.knowledge_graph

    # from src.core.logging_config import setup_logging # Explicitly call if not auto-setup
    # setup_logging()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    logger.info("--- Testing KnowledgeGraphService ---")
    kb = KnowledgeBaseService() # Load data into KB
    kg_service = KnowledgeGraphService(kb_service=kb) # Populate KG from KB

    # Test node and edge counts after population
    logger.info(f"Graph nodes after population: {kg_service.graph.number_of_nodes()}, Edges: {kg_service.graph.number_of_edges()}")

    # Verify some new nodes (e.g., a financial statement and a default event)
    sample_fs_id = "FS001" # From sample_financial_statements.json
    fs_node_attrs = kg_service.get_node_attributes(sample_fs_id)
    if fs_node_attrs:
        logger.info(f"Found FinancialStatement node {sample_fs_id}, type: {fs_node_attrs.get('type')}, revenue: {fs_node_attrs.get('revenue')}")
    else:
        logger.warning(f"FinancialStatement node {sample_fs_id} not found.")

    sample_de_id = "DE001" # From sample_default_events.json
    de_node_attrs = kg_service.get_node_attributes(sample_de_id)
    if de_node_attrs:
        logger.info(f"Found DefaultEvent node {sample_de_id}, type: {de_node_attrs.get('type')}, default_type: {de_node_attrs.get('default_type')}")
    else:
        logger.warning(f"DefaultEvent node {sample_de_id} not found.")

    # Test get_node_attributes for a company and a loan
    test_company_id = "COMP001" # Has subsidiaries, loans, FS
    node_attrs = kg_service.get_node_attributes(test_company_id)
    if node_attrs:
        logger.info(f"Attributes for {test_company_id}: Name: {node_attrs.get('company_name')}, Subsidiaries: {node_attrs.get('subsidiaries')}")

    test_loan_id = "LOAN7001" # Has guarantors in sample data
    loan_attrs = kg_service.get_node_attributes(test_loan_id)
    if loan_attrs:
        logger.info(f"Attributes for {test_loan_id}: Amount: {loan_attrs.get('loan_amount')}, Guarantors: {loan_attrs.get('guarantors')}")


    # Test find_common_guarantors
    logger.info("--- Testing find_common_guarantors ---")
    # Assuming LOAN7001 is guaranteed by COMP002, and we add another loan also guaranteed by COMP002
    # For test, let's assume LOAN7001 and LOAN7004 (if it had COMP002 as guarantor)
    # From sample_loans.json: LOAN7001 guaranteed by COMP002. LOAN7004 by COMP001.
    # Let's test with LOAN7001 and a hypothetical LOAN_TEST also guaranteed by COMP002
    # kg_service.add_node("LOAN_TEST", node_type="LoanAgreement", company_id="COMP_TEST")
    # kg_service.add_node("COMP002", node_type="CorporateEntity") # Ensure guarantor node exists
    # kg_service.add_edge("COMP002", "LOAN_TEST", RelationshipType.GUARANTEES)
    # kg_service.add_edge("LOAN_TEST", "COMP002", RelationshipType.HAS_GUARANTOR)
    # common_guarantors = kg_service.find_common_guarantors(["LOAN7001", "LOAN_TEST"])
    # logger.info(f"Common guarantors for LOAN7001, LOAN_TEST: {common_guarantors}")
    # For now, using existing data:
    # LOAN7001 has COMP002; LOAN7004 has COMP001 - no common ones by default.
    # Let's manually create a scenario in the test if needed or use loans that WILL have common guarantors from data.
    # Sample data: LOAN7001 by COMP002. LOAN7004 by COMP001. No common.
    # If we had LOAN_X by COMP002, then find_common_guarantors(["LOAN7001", "LOAN_X"]) would yield {"COMP002": ["LOAN7001", "LOAN_X"]}
    # The current sample data does not have common guarantors for multiple loans.
    # We will test with loan IDs that do not share guarantors to see empty result.
    common_guarantors_test = kg_service.find_common_guarantors(["LOAN7001", "LOAN7004"])
    logger.info(f"Common guarantors for LOAN7001, LOAN7004: {common_guarantors_test}") # Expected: {}

    # Test get_entity_centrality
    logger.info("--- Testing get_entity_centrality ---")
    for alg in ['degree', 'betweenness', 'closeness']:
        centrality = kg_service.get_entity_centrality(test_company_id, algorithm=alg)
        logger.info(f"{alg.capitalize()} centrality for {test_company_id}: {centrality:.4f}")
    centrality_non_existent = kg_service.get_entity_centrality("NON_EXISTENT_ID", algorithm='degree')
    logger.info(f"Degree centrality for NON_EXISTENT_ID: {centrality_non_existent}")


    # Test find_paths_between_entities
    logger.info("--- Testing find_paths_between_entities ---")
    # Path from company to its financial statement
    paths_to_fs = kg_service.find_paths_between_entities(test_company_id, sample_fs_id, cutoff=1)
    logger.info(f"Paths from {test_company_id} to {sample_fs_id} (direct): {paths_to_fs}")

    # Path from company to one of its loan's default events
    # COMP003 -> LOAN7004 -> DE001
    defaulting_company_id = "COMP003" # Has LOAN7004 which has DE001
    default_event_for_comp3_loan = "DE001"
    if kg_service.graph.has_node(defaulting_company_id) and kg_service.graph.has_node(default_event_for_comp3_loan):
        paths_to_de = kg_service.find_paths_between_entities(defaulting_company_id, default_event_for_comp3_loan, cutoff=3)
        logger.info(f"Paths from {defaulting_company_id} to {default_event_for_comp3_loan} (max 3 hops): {paths_to_de}")
        # Filtered by relationship type
        # This path involves HAS_LOAN, HAS_DEFAULT_EVENT. Let's test with HAS_LOAN.
        paths_to_de_filtered = kg_service.find_paths_between_entities(
            defaulting_company_id, default_event_for_comp3_loan,
            relationship_types=[RelationshipType.HAS_LOAN, RelationshipType.HAS_DEFAULT_EVENT], cutoff=3
        )
        logger.info(f"Paths from {defaulting_company_id} to {default_event_for_comp3_loan} (filtered by HAS_LOAN, HAS_DEFAULT_EVENT): {paths_to_de_filtered}")
    else:
        logger.warning(f"Could not test paths to default event for {defaulting_company_id} as nodes are missing.")

    # Test get_company_contextual_info
    logger.info("--- Testing get_company_contextual_info ---")
    context_info_comp1 = kg_service.get_company_contextual_info(test_company_id)
    logger.info(f"Contextual info for {test_company_id}: {context_info_comp1}")

    context_info_comp4 = kg_service.get_company_contextual_info(defaulting_company_id) # COMP003 actually, was COMP004
    logger.info(f"Contextual info for {defaulting_company_id}: {context_info_comp4}")

    context_info_non_existent = kg_service.get_company_contextual_info("NON_EXISTENT_ID")
    logger.info(f"Contextual info for NON_EXISTENT_ID: {context_info_non_existent}")

    # Verify specific relationships for COMP001
    comp1_subs = kg_service.get_neighbors(test_company_id, relationship_type=RelationshipType.HAS_SUBSIDIARY)
    logger.info(f"COMP001 HAS_SUBSIDIARY: {comp1_subs}")
    comp1_fs = kg_service.get_neighbors(test_company_id, relationship_type=RelationshipType.HAS_FINANCIAL_STATEMENT)
    logger.info(f"COMP001 HAS_FINANCIAL_STATEMENT: {comp1_fs}")

    # Verify guarantor relationship for LOAN7001
    loan1_guarantors = kg_service.get_neighbors(test_loan_id, relationship_type=RelationshipType.HAS_GUARANTOR)
    logger.info(f"LOAN7001 HAS_GUARANTOR: {loan1_guarantors}")
    if loan1_guarantors:
        guarantor_loans = kg_service.get_neighbors(loan1_guarantors[0], relationship_type=RelationshipType.GUARANTEES)
        logger.info(f"{loan1_guarantors[0]} GUARANTEES: {guarantor_loans}")

    # Verify default event for LOAN7004
    loan4_de = kg_service.get_neighbors("LOAN7004", relationship_type=RelationshipType.HAS_DEFAULT_EVENT)
    logger.info(f"LOAN7004 HAS_DEFAULT_EVENT: {loan4_de}")
