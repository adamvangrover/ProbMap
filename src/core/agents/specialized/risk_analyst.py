import logging
from typing import Dict, Any, Optional
from src.core.agents.base import BaseAgent
from src.core.agents.protocol import AgentMessage, MessageType
from src.risk_models.bayesian import BayesianCreditModel

logger = logging.getLogger(__name__)

class RiskAnalystAgent(BaseAgent):
    def __init__(self, agent_id: str, company_id: Optional[str] = None, prior_probs: Dict[str, float] = None, registry: Any = None):
        super().__init__(agent_id)
        self.model = BayesianCreditModel(prior_probs=prior_probs)
        self.state.memory["belief_distribution"] = self.model.get_belief_distribution()
        self.registry = registry
        self.company_id = company_id
        if company_id:
            self.state.memory["company_id"] = company_id

    async def process_message(self, message: AgentMessage) -> None:
        if message.message_type == MessageType.TASK and isinstance(message.content, dict) and "evidence" in message.content:
            evidence = message.content["evidence"]
            evidence_type = evidence.get("type")
            strength = evidence.get("strength", "medium")

            logger.info(f"RiskAnalyst {self.agent_id} (Company: {self.company_id}) processing evidence: {evidence_type}")

            self.model.update(evidence_type, strength)

            new_pd = self.model.get_expected_pd()
            new_dist = self.model.get_belief_distribution()

            self.state.memory["belief_distribution"] = new_dist
            self.state.memory["expected_pd"] = new_pd

            logger.info(f"RiskAnalyst {self.agent_id} updated PD: {new_pd}")

            if self.registry:
                sender_agent = self.registry.get_agent(message.sender)
                if sender_agent:
                    response_content = {
                        "pd": new_pd,
                        "rating": self.model.get_most_likely_rating(),
                        "company_id": self.company_id
                    }
                    await self.send_message(
                        sender_agent,
                        content=response_content,
                        message_type=MessageType.RESPONSE,
                        metadata={"reply_to": message.id}
                    )

        elif message.message_type == MessageType.REQUEST and isinstance(message.content, dict) and message.content.get("action") == "get_pd":
            if self.registry:
                sender_agent = self.registry.get_agent(message.sender)
                if sender_agent:
                    response_content = {
                        "pd": self.model.get_expected_pd(),
                        "rating": self.model.get_most_likely_rating(),
                        "company_id": self.company_id
                    }
                    await self.send_message(
                        sender_agent,
                        content=response_content,
                        message_type=MessageType.RESPONSE,
                        metadata={"reply_to": message.id}
                    )
