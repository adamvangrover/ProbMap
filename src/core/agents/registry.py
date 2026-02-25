from typing import Dict, Optional
from src.core.agents.base import BaseAgent

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent):
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent with ID {agent.agent_id} already registered.")
        self._agents[agent.agent_id] = agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_id)

    def remove_agent(self, agent_id: str):
        if agent_id in self._agents:
            del self._agents[agent_id]

    def list_agents(self) -> Dict[str, BaseAgent]:
        return self._agents
