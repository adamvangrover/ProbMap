import logging
import numpy as np
from typing import Any, Optional
from src.core.agents.base import BaseAgent
from src.core.agents.protocol import AgentMessage, MessageType
from src.simulation.oswm import OneShotWorldModel

logger = logging.getLogger(__name__)

class ForecasterAgent(BaseAgent):
    def __init__(self, agent_id: str, model_path: Optional[str] = None, input_dim: int = 10, registry: Any = None):
        super().__init__(agent_id)
        self.model = OneShotWorldModel(input_dim=input_dim)
        if model_path:
            self.model.load_model(model_path)
        self.registry = registry

    async def process_message(self, message: AgentMessage) -> None:
        if message.message_type == MessageType.TASK and isinstance(message.content, dict) and "context_sequence" in message.content:
            context_seq = message.content["context_sequence"]
            prediction_steps = message.content.get("prediction_steps", 1)

            logger.info(f"Forecaster {self.agent_id} predicting for {prediction_steps} steps.")

            # Ensure context_seq is np.ndarray
            try:
                context_arr = np.array(context_seq)
                predictions = self.model.adapt_and_predict(context_arr, prediction_steps=prediction_steps)

                # Convert predictions back to list for JSON serialization
                predictions_list = predictions.tolist()

                if self.registry:
                    sender_agent = self.registry.get_agent(message.sender)
                    if sender_agent:
                        await self.send_message(
                            sender_agent,
                            content={"predictions": predictions_list},
                            message_type=MessageType.RESPONSE,
                            metadata={"reply_to": message.id}
                        )
            except Exception as e:
                logger.error(f"Forecaster prediction error: {e}")
                if self.registry:
                    sender_agent = self.registry.get_agent(message.sender)
                    if sender_agent:
                        await self.send_message(
                            sender_agent,
                            content={"error": str(e)},
                            message_type=MessageType.ERROR,
                            metadata={"reply_to": message.id}
                        )
