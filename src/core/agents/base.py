import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime

from src.core.agents.protocol import AgentMessage, AgentState, MessageType

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.inbox = asyncio.Queue()
        self.state = AgentState(agent_id=agent_id)

    async def send_message(self, receiver: 'BaseAgent', content: Any, message_type: MessageType = MessageType.INFO, metadata: Dict = None) -> None:
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver.agent_id,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        await receiver.receive_message(message)
        logger.debug(f"Agent {self.agent_id} sent message to {receiver.agent_id}: {message.id}")

    async def receive_message(self, message: AgentMessage) -> None:
        await self.inbox.put(message)
        logger.debug(f"Agent {self.agent_id} received message: {message.id}")

    async def process_inbox(self) -> None:
        """
        Process all messages currently in the inbox.
        """
        while not self.inbox.empty():
            message = await self.inbox.get()
            try:
                await self.process_message(message)
            except Exception as e:
                logger.error(f"Error processing message {message.id} in agent {self.agent_id}: {e}")
            finally:
                self.inbox.task_done()

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> None:
        """Subclasses must implement this to handle messages."""
        pass

    def update_status(self, status: str):
        self.state.status = status
        self.state.last_active = datetime.utcnow()
