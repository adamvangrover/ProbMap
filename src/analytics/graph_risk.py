import logging
import networkx as nx
import pandas as pd
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class GraphRiskEngine:
    """
    Analyzes the Knowledge Graph to calculate network-based risk metrics.
    Focuses on contagion, supply chain vulnerability, and systemic importance.
    """
    def __init__(self, knowledge_graph_service):
        self.kg_service = knowledge_graph_service
        # Access the underlying NetworkX graph directly for complex algorithms
        self.graph = self.kg_service.graph

    def calculate_contagion_score(self, company_id: str, damping_factor: float = 0.85) -> float:
        """
        Calculates a 'Contagion Score' representing the company's susceptibility
        to risk propagating through the network.

        Uses a variation of PageRank where 'votes' are riskiness of neighbors.
        """
        if not self.graph.has_node(company_id):
            return 0.0

        # Simplified: weighted sum of neighbor degrees (systemic importance of neighbors)
        # In a real system, this would use neighbor PDs as weights.
        score = 0.0
        try:
            # 1. Direct neighbors impact
            neighbors = list(self.graph.neighbors(company_id))
            for neighbor in neighbors:
                # Retrieve relationship type if possible (simplified here)
                # edge_data = self.graph.get_edge_data(company_id, neighbor)

                # neighbor importance (degree)
                deg = self.graph.degree[neighbor]
                score += deg

            # Normalize score
            score = score / (len(neighbors) + 1) if neighbors else 0.0

            # 2. Centrality based adjustment
            # Calculate local centrality only to save compute?
            # Or use pre-calculated centrality if available in node attributes.
            # Assuming small graph for PoC, we calculate full PageRank
            pr = nx.pagerank(self.graph, alpha=damping_factor)
            centrality = pr.get(company_id, 0.0)

            # Combined score: Susceptibility (neighbors) * Importance (Centrality)
            final_score = score * (1 + centrality)
            return round(final_score, 4)

        except Exception as e:
            logger.error(f"Error calculating contagion score for {company_id}: {e}")
            return 0.0

    def calculate_supply_chain_risk(self, company_id: str) -> Dict[str, Any]:
        """
        Analyzes the upstream (suppliers) and downstream (customers) risks.
        """
        if not self.graph.has_node(company_id):
            return {}

        suppliers = []
        customers = []

        # In the KG setup (from knowledge_graph.py), we assume:
        # (Company) -[HAS_SUPPLIER]-> (Supplier)
        # (Company) -[HAS_CUSTOMER]-> (Customer)
        # We need to traverse edges and check types.

        try:
            out_edges = self.graph.out_edges(company_id, data=True)
            for _, target, data in out_edges:
                rel_type = data.get('relation')
                if rel_type == "HAS_SUPPLIER":
                    suppliers.append(target)
                elif rel_type == "HAS_CUSTOMER":
                    customers.append(target)

            # Vulnerability: Ratio of suppliers to customers (dependency)
            # High suppliers, low customers -> High dependency, but maybe robust supply?
            # Low suppliers, high customers -> Single point of failure in supply?
            # Metric: Supplier Concentration Risk = 1 / (Num Suppliers + 1)

            concentration_risk = 1.0 / (len(suppliers) + 1)

            return {
                "num_suppliers": len(suppliers),
                "num_customers": len(customers),
                "supplier_ids": suppliers,
                "customer_ids": customers,
                "supply_chain_concentration_risk": round(concentration_risk, 4)
            }
        except Exception as e:
            logger.error(f"Error calculating supply chain risk for {company_id}: {e}")
            return {}

    def get_network_stats(self, company_id: str) -> Dict[str, Any]:
        """
        Aggregates all graph metrics for a company.
        """
        return {
            "contagion_score": self.calculate_contagion_score(company_id),
            "supply_chain_analysis": self.calculate_supply_chain_risk(company_id),
            "degree_centrality": round(nx.degree_centrality(self.graph).get(company_id, 0), 4) if self.graph else 0
        }
