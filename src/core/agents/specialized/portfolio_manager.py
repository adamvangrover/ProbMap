import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.core.agents.base import BaseAgent
from src.core.agents.protocol import AgentMessage, MessageType
from src.core.agents.specialized.risk_analyst import RiskAnalystAgent
from src.data_management.knowledge_base import KnowledgeBaseService
from src.risk_map.visualization import RiskVisualizationService

logger = logging.getLogger(__name__)

class PortfolioManagerAgent(BaseAgent):
    def __init__(self, agent_id: str, kb_service: KnowledgeBaseService, registry: Any):
        super().__init__(agent_id)
        self.kb_service = kb_service
        self.registry = registry
        self.visualization_service = RiskVisualizationService()
        self.company_agents: Dict[str, str] = {} # Map company_id -> agent_id
        self.risk_map_state: Dict[str, Dict[str, Any]] = {} # Map company_id -> latest risk data
        self.risk_history: List[Dict[str, Any]] = [] # Time series of risk updates

    async def initialize_risk_agents(self):
        """
        Scans the KB and creates a RiskAnalystAgent for each company.
        """
        companies = self.kb_service.get_all_companies()
        logger.info(f"PortfolioManager initializing agents for {len(companies)} companies.")

        for company in companies:
            agent_id = f"risk_analyst_{company.company_id}"

            # Create agent with initial belief based on data if available
            # For now we just init with default, but could pull from RiskMapService
            risk_agent = RiskAnalystAgent(agent_id=agent_id, company_id=company.company_id, registry=self.registry)

            self.registry.register_agent(risk_agent)
            self.company_agents[company.company_id] = agent_id
            self.risk_map_state[company.company_id] = {
                "status": "INITIALIZED",
                "pd": None,
                "industry_sector": company.industry_sector.value if company.industry_sector else "Unknown"
            }

            logger.info(f"Created agent {agent_id} for company {company.company_name}")

    async def process_message(self, message: AgentMessage) -> None:
        if message.message_type == MessageType.TASK and isinstance(message.content, dict):
            # Handle Market Event routing
            if "event_type" in message.content and "company_id" in message.content:
                target_company = message.content["company_id"]
                if target_company in self.company_agents:
                    target_agent_id = self.company_agents[target_company]
                    target_agent = self.registry.get_agent(target_agent_id)

                    if target_agent:
                        logger.info(f"PortfolioManager routing event for {target_company} to {target_agent_id}")

                        # Forward the evidence
                        evidence_content = {
                            "evidence": {
                                "type": message.content["event_type"],
                                "strength": message.content.get("strength", "medium")
                            }
                        }

                        await self.send_message(
                            target_agent,
                            content=evidence_content,
                            message_type=MessageType.TASK,
                            metadata={"forwarded_from": message.sender}
                        )
                else:
                    logger.warning(f"PortfolioManager received event for unknown company: {target_company}")

        elif message.message_type == MessageType.RESPONSE and isinstance(message.content, dict):
            # Handle Risk Update from Analyst
            if "pd" in message.content:
                sender_id = message.sender
                # Reverse lookup company from agent_id (simplified)
                # In a real app we'd pass company_id in metadata
                company_id = self._get_company_for_agent(sender_id)
                if company_id:
                    new_pd = message.content["pd"]
                    rating = message.content.get("rating")

                    self.risk_map_state[company_id]["pd"] = new_pd
                    self.risk_map_state[company_id]["rating"] = rating
                    self.risk_map_state[company_id]["status"] = "UPDATED"

                    # Record history
                    sector = self.risk_map_state[company_id].get("industry_sector", "Unknown")
                    history_entry = {
                        "timestamp": datetime.now(),
                        "company_id": company_id,
                        "industry_sector": sector,
                        "pd": new_pd
                    }
                    self.risk_history.append(history_entry)

                    logger.info(f"PortfolioManager updated Risk Map for {company_id}: PD={new_pd}, Rating={rating}")

    def _get_company_for_agent(self, agent_id: str) -> Optional[str]:
        for cid, aid in self.company_agents.items():
            if aid == agent_id:
                return cid
        return None

    def generate_report(self, output_path: str = "output/risk_trajectory.png"):
        logger.info(f"Generating risk trajectory report to {output_path}")
        self.visualization_service.generate_risk_trajectory_plot(self.risk_history, output_path=output_path)
