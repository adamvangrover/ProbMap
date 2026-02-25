import asyncio
import logging
import random
import numpy as np
import os
from typing import List, Dict, Any

from src.core.agents.registry import AgentRegistry
from src.core.agents.specialized.portfolio_manager import PortfolioManagerAgent
from src.core.agents.specialized.forecaster import ForecasterAgent
from src.core.agents.base import BaseAgent
from src.core.agents.protocol import AgentMessage, MessageType
from src.data_management.knowledge_base import KnowledgeBaseService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketAgent(BaseAgent):
    """
    A simple agent that simulates market events.
    """
    def __init__(self, agent_id: str, registry: AgentRegistry):
        super().__init__(agent_id)
        self.registry = registry

    async def process_message(self, message: AgentMessage) -> None:
        logger.info(f"MarketAgent {self.agent_id} received: {message.content}")

    async def broadcast_event(self, company_id: str, event_type: str, strength: str):
        # Broadcast to Portfolio Manager
        pm = self.registry.get_agent("portfolio_manager_01")
        if pm:
            logger.info(f"MarketAgent broadcasting event for {company_id}: {event_type}")
            await self.send_message(
                pm,
                content={"event_type": event_type, "strength": strength, "company_id": company_id},
                message_type=MessageType.TASK
            )

    async def request_forecast(self):
        forecaster = self.registry.get_agent("forecaster_01")
        if forecaster:
            # Generate random context sequence
            context = np.random.randn(5, 10).tolist()
            logger.info("MarketAgent requesting forecast")
            await self.send_message(
                forecaster,
                content={"context_sequence": context, "prediction_steps": 2},
                message_type=MessageType.TASK
            )

async def run_simulation(duration_seconds: int = 5):
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    registry = AgentRegistry()

    # Initialize KB
    kb_service = KnowledgeBaseService()

    # Create Portfolio Manager
    portfolio_manager = PortfolioManagerAgent("portfolio_manager_01", kb_service=kb_service, registry=registry)
    registry.register_agent(portfolio_manager)

    # Initialize Risk Agents for Companies in KB
    await portfolio_manager.initialize_risk_agents()

    # Create other agents
    forecaster = ForecasterAgent("forecaster_01", input_dim=10, registry=registry)
    market_agent = MarketAgent("market_01", registry=registry)

    registry.register_agent(forecaster)
    registry.register_agent(market_agent)

    logger.info("Simulation started.")

    loop = asyncio.get_running_loop()
    end_time = loop.time() + duration_seconds

    # Scenario: A series of events affecting different companies over time
    events = [
        (0.5, "COMP001", "positive_earnings", "high"),
        (1.0, "COMP002", "lawsuit", "medium"),
        (1.5, "COMP001", "new_contract", "medium"),
        (2.0, "COMP003", "downgrade", "high"),
        (2.5, "COMP002", "missed_payment", "medium"),
        (3.0, "COMP004", "management_churn", "low"),
        (3.5, "COMP001", "upgrade", "low"),
    ]

    event_idx = 0
    start_time = loop.time()

    while loop.time() < end_time:
        current_time_offset = loop.time() - start_time

        # Trigger scheduled events
        if event_idx < len(events):
            trigger_time, cid, etype, strength = events[event_idx]
            if current_time_offset >= trigger_time:
                await market_agent.broadcast_event(cid, etype, strength)
                event_idx += 1

        # Process inboxes
        for agent in registry.list_agents().values():
            await agent.process_inbox()

        await asyncio.sleep(0.1)

    logger.info("Simulation ended.")

    # Log final state of Risk Map
    logger.info("Final Risk Map State:")
    for cid, state in portfolio_manager.risk_map_state.items():
        if state["status"] == "UPDATED":
            logger.info(f"  Company {cid}: PD={state.get('pd')}, Rating={state.get('rating')}")

    # Generate Visualization
    portfolio_manager.generate_report("output/risk_trajectory.png")
    logger.info("Risk trajectory report generated at output/risk_trajectory.png")

if __name__ == "__main__":
    asyncio.run(run_simulation())
