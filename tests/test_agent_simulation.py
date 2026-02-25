import sys
from unittest.mock import MagicMock

# Mock torch before it is imported by oswm
mock_torch = MagicMock()
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.optim"] = MagicMock()

import pytest
import asyncio
import numpy as np
from src.core.agents.registry import AgentRegistry
from src.core.agents.specialized.risk_analyst import RiskAnalystAgent
from src.core.agents.specialized.forecaster import ForecasterAgent
from src.core.agents.specialized.portfolio_manager import PortfolioManagerAgent
from src.core.agents.base import BaseAgent
from src.core.agents.protocol import AgentMessage, MessageType
from src.data_management.ontology import CorporateEntity, IndustrySector

class MockAgent(BaseAgent):
    def __init__(self, agent_id: str, registry: AgentRegistry):
        super().__init__(agent_id)
        self.registry = registry
        self.received_messages = []

    async def process_message(self, message: AgentMessage) -> None:
        self.received_messages.append(message)

@pytest.mark.asyncio
async def test_risk_analyst_interaction():
    registry = AgentRegistry()

    # Create agents
    risk_analyst = RiskAnalystAgent("risk_agent", company_id="TEST_COMP", registry=registry)
    mock_sender = MockAgent("sender_agent", registry=registry)

    registry.register_agent(risk_analyst)
    registry.register_agent(mock_sender)

    # Send evidence message
    await mock_sender.send_message(
        risk_analyst,
        content={"evidence": {"type": "positive_earnings", "strength": "high"}},
        message_type=MessageType.TASK
    )

    # Process risk analyst inbox
    await risk_analyst.process_inbox()

    # Check if risk analyst updated state
    assert "belief_distribution" in risk_analyst.state.memory
    assert risk_analyst.state.memory["company_id"] == "TEST_COMP"

    # Process sender inbox (to check for reply)
    await mock_sender.process_inbox()

    assert len(mock_sender.received_messages) == 1
    reply = mock_sender.received_messages[0]
    assert reply.sender == "risk_agent"
    assert "pd" in reply.content
    assert reply.content["company_id"] == "TEST_COMP"

@pytest.mark.asyncio
async def test_forecaster_interaction():
    registry = AgentRegistry()

    forecaster = ForecasterAgent("forecast_agent", input_dim=5, registry=registry)
    forecaster.model.adapt_and_predict = MagicMock(return_value=np.zeros((2, 5)))

    mock_sender = MockAgent("sender_agent", registry=registry)

    registry.register_agent(forecaster)
    registry.register_agent(mock_sender)

    # Send forecast request
    context = np.random.randn(3, 5).tolist()
    await mock_sender.send_message(
        forecaster,
        content={"context_sequence": context, "prediction_steps": 2},
        message_type=MessageType.TASK
    )

    # Process
    await forecaster.process_inbox()
    await mock_sender.process_inbox()

    # Verify reply
    assert len(mock_sender.received_messages) == 1
    reply = mock_sender.received_messages[0]
    assert "predictions" in reply.content

@pytest.mark.asyncio
async def test_portfolio_manager_routing():
    registry = AgentRegistry()

    # Mock KB Service
    mock_kb = MagicMock()
    mock_company = CorporateEntity(
        company_id="COMP001",
        company_name="Test Corp",
        industry_sector=IndustrySector.TECHNOLOGY,
        country_iso_code="US"
    )
    mock_kb.get_all_companies.return_value = [mock_company]

    # Create Portfolio Manager
    pm = PortfolioManagerAgent("pm_agent", kb_service=mock_kb, registry=registry)
    registry.register_agent(pm)

    # Initialize agents
    await pm.initialize_risk_agents()

    # Verify agent creation
    assert "COMP001" in pm.company_agents
    risk_agent_id = pm.company_agents["COMP001"]
    assert registry.get_agent(risk_agent_id) is not None

    # Simulate Market Event
    mock_market = MockAgent("market_agent", registry=registry)
    registry.register_agent(mock_market)

    await mock_market.send_message(
        pm,
        content={"event_type": "lawsuit", "strength": "medium", "company_id": "COMP001"},
        message_type=MessageType.TASK
    )

    # Process PM inbox (routes to Risk Agent)
    await pm.process_inbox()

    # Process Risk Agent inbox (processes evidence)
    risk_agent = registry.get_agent(risk_agent_id)
    await risk_agent.process_inbox()

    # Process PM inbox (receives update from Risk Agent)
    await pm.process_inbox()

    # Check PM state
    assert pm.risk_map_state["COMP001"]["status"] == "UPDATED"
    assert pm.risk_map_state["COMP001"]["pd"] is not None
