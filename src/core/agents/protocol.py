import uuid
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    INFO = "INFO"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    ERROR = "ERROR"
    TASK = "TASK"

class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the message")
    sender: str = Field(..., description="ID of the sender agent")
    receiver: str = Field(..., description="ID of the receiver agent")
    content: Any = Field(..., description="The content of the message")
    message_type: MessageType = Field(default=MessageType.INFO, description="Type of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Time of creation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentState(BaseModel):
    agent_id: str
    status: str = "IDLE"
    memory: Dict[str, Any] = Field(default_factory=dict)
    last_active: datetime = Field(default_factory=datetime.utcnow)
