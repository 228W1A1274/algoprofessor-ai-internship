"""
Communication Protocol - Day 9 Multi-Agent System
Handles all inter-agent communication and message routing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import json
import threading
from queue import Queue
from collections import defaultdict


class MessageType(Enum):
    """Types of messages agents can send"""
    TASK_ASSIGNMENT = "task_assignment"
    DATA_TRANSFER = "data_transfer"
    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ERROR = "error"
    STATUS_UPDATE = "status_update"
    COMPLETION = "completion"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """
    Standardized message format for inter-agent communication
    
    Attributes:
        from_agent: Name of sending agent
        to_agent: Name of receiving agent
        message_type: Type of message
        content: Message payload
        priority: Message priority
        requires_response: Whether sender expects a response
        correlation_id: ID to track related messages
        metadata: Additional message metadata
    """
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Any
    priority: MessagePriority = MessagePriority.NORMAL
    requires_response: bool = False
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    message_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.message_type.value,
            "content": self.content,
            "priority": self.priority.value,
            "requires_response": self.requires_response,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def __str__(self) -> str:
        """String representation of message"""
        return f"[{self.message_type.value.upper()}] {self.from_agent} → {self.to_agent}"


class MessageBus:
    """
    Central message bus for routing messages between agents
    
    Features:
    - Message queuing
    - Priority handling
    - Message history tracking
    - Dead letter queue for failed messages
    - Thread-safe operations
    """
    
    def __init__(self, max_history: int = 1000):
        self.queues: Dict[str, Queue] = defaultdict(Queue)
        self.message_history: List[Message] = []
        self.dead_letter_queue: List[Message] = []
        self.max_history = max_history
        self.lock = threading.Lock()
        self.stats = {
            "total_sent": 0,
            "total_delivered": 0,
            "total_failed": 0,
            "by_type": defaultdict(int),
            "by_agent": defaultdict(lambda: {"sent": 0, "received": 0})
        }
    
    def send(self, message: Message) -> bool:
        """
        Send a message to an agent
        
        Args:
            message: Message to send
        
        Returns:
            True if message was queued successfully
        """
        try:
            with self.lock:
                # Add to recipient's queue
                self.queues[message.to_agent].put(message)
                
                # Update statistics
                self.stats["total_sent"] += 1
                self.stats["by_type"][message.message_type.value] += 1
                self.stats["by_agent"][message.from_agent]["sent"] += 1
                self.stats["by_agent"][message.to_agent]["received"] += 1
                
                # Add to history (maintain max size)
                self.message_history.append(message)
                if len(self.message_history) > self.max_history:
                    self.message_history.pop(0)
                
                # Log message
                self._log_message(message, "SENT")
                
                return True
                
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            self.dead_letter_queue.append(message)
            self.stats["total_failed"] += 1
            return False
    
    def receive(self, agent_name: str, timeout: float = 1.0) -> Optional[Message]:
        """
        Receive a message for an agent
        
        Args:
            agent_name: Name of agent receiving message
            timeout: How long to wait for a message
        
        Returns:
            Message if available, None otherwise
        """
        try:
            message = self.queues[agent_name].get(timeout=timeout)
            self.stats["total_delivered"] += 1
            self._log_message(message, "DELIVERED")
            return message
        except:
            return None
    
    def has_messages(self, agent_name: str) -> bool:
        """Check if agent has pending messages"""
        return not self.queues[agent_name].empty()
    
    def get_pending_count(self, agent_name: str) -> int:
        """Get number of pending messages for an agent"""
        return self.queues[agent_name].qsize()
    
    def _log_message(self, message: Message, status: str) -> None:
        """Log message with emoji indicator"""
        emoji = {
            "SENT": "📤",
            "DELIVERED": "📨",
            "FAILED": "❌"
        }.get(status, "📋")
        
        print(f"{emoji} [{status}] {message}")
    
    def get_conversation(self, agent1: str, agent2: str) -> List[Message]:
        """
        Get all messages between two agents
        
        Args:
            agent1: First agent name
            agent2: Second agent name
        
        Returns:
            List of messages between the agents
        """
        return [
            msg for msg in self.message_history
            if (msg.from_agent == agent1 and msg.to_agent == agent2) or
               (msg.from_agent == agent2 and msg.to_agent == agent1)
        ]
    
    def get_agent_messages(self, agent_name: str, sent: bool = True, received: bool = True) -> List[Message]:
        """
        Get all messages for an agent
        
        Args:
            agent_name: Agent name
            sent: Include sent messages
            received: Include received messages
        
        Returns:
            List of messages
        """
        messages = []
        for msg in self.message_history:
            if sent and msg.from_agent == agent_name:
                messages.append(msg)
            if received and msg.to_agent == agent_name:
                messages.append(msg)
        return messages
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            "total_messages_sent": self.stats["total_sent"],
            "total_messages_delivered": self.stats["total_delivered"],
            "total_messages_failed": self.stats["total_failed"],
            "messages_by_type": dict(self.stats["by_type"]),
            "messages_by_agent": dict(self.stats["by_agent"]),
            "messages_in_history": len(self.message_history),
            "messages_in_dead_letter": len(self.dead_letter_queue)
        }
    
    def export_history(self, filepath: str) -> None:
        """
        Export message history to JSON file
        
        Args:
            filepath: Path to save history
        """
        history_data = {
            "exported_at": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "messages": [msg.to_dict() for msg in self.message_history],
            "failed_messages": [msg.to_dict() for msg in self.dead_letter_queue]
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"✅ Message history exported to {filepath}")
    
    def clear_history(self) -> None:
        """Clear message history"""
        with self.lock:
            self.message_history.clear()
            self.dead_letter_queue.clear()
            print("🗑️ Message history cleared")


class ConversationManager:
    """
    Manages conversations between agents with correlation tracking
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.conversations: Dict[str, List[Message]] = {}
    
    def start_conversation(self, conversation_id: str, initiator: str, recipient: str, topic: str) -> str:
        """
        Start a new conversation
        
        Args:
            conversation_id: Unique conversation ID
            initiator: Agent starting the conversation
            recipient: Agent receiving the conversation
            topic: Conversation topic
        
        Returns:
            Conversation ID
        """
        self.conversations[conversation_id] = []
        
        initial_message = Message(
            from_agent=initiator,
            to_agent=recipient,
            message_type=MessageType.REQUEST,
            content={"topic": topic, "conversation_started": True},
            correlation_id=conversation_id,
            requires_response=True
        )
        
        self.message_bus.send(initial_message)
        self.conversations[conversation_id].append(initial_message)
        
        print(f"💬 Started conversation '{conversation_id}': {initiator} → {recipient}")
        return conversation_id
    
    def add_to_conversation(self, conversation_id: str, message: Message) -> None:
        """Add a message to an existing conversation"""
        if conversation_id in self.conversations:
            message.correlation_id = conversation_id
            self.conversations[conversation_id].append(message)
            self.message_bus.send(message)
    
    def get_conversation(self, conversation_id: str) -> List[Message]:
        """Get all messages in a conversation"""
        return self.conversations.get(conversation_id, [])
    
    def end_conversation(self, conversation_id: str, final_agent: str) -> None:
        """Mark a conversation as ended"""
        if conversation_id in self.conversations:
            print(f"✅ Conversation '{conversation_id}' ended by {final_agent}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_task_message(from_agent: str, to_agent: str, task_description: str, task_data: Any = None) -> Message:
    """Create a task assignment message"""
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=MessageType.TASK_ASSIGNMENT,
        content={
            "description": task_description,
            "data": task_data
        },
        priority=MessagePriority.HIGH,
        requires_response=True
    )


def create_data_message(from_agent: str, to_agent: str, data: Any, description: str = "") -> Message:
    """Create a data transfer message"""
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=MessageType.DATA_TRANSFER,
        content={
            "data": data,
            "description": description
        },
        priority=MessagePriority.NORMAL
    )


def create_feedback_message(from_agent: str, to_agent: str, feedback: str, suggestions: List[str] = None) -> Message:
    """Create a feedback message"""
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=MessageType.FEEDBACK,
        content={
            "feedback": feedback,
            "suggestions": suggestions or []
        },
        priority=MessagePriority.HIGH,
        requires_response=True
    )


def create_completion_message(from_agent: str, to_agent: str, result: Any, success: bool = True) -> Message:
    """Create a task completion message"""
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type=MessageType.COMPLETION,
        content={
            "result": result,
            "success": success
        },
        priority=MessagePriority.HIGH
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import os
    
    print("\n📡 COMMUNICATION PROTOCOL - Testing\n")
    print("="*60)
    
    # Create message bus
    bus = MessageBus()
    
    # Test 1: Simple message sending
    print("\n1️⃣ Testing Simple Message Sending...\n")
    
    msg1 = create_task_message(
        from_agent="Orchestrator",
        to_agent="Researcher",
        task_description="Research AI trends in 2025"
    )
    bus.send(msg1)
    
    # Test 2: Receiving messages
    print("\n2️⃣ Testing Message Receiving...\n")
    
    received = bus.receive("Researcher", timeout=0.1)
    if received:
        print(f"✅ Received: {received}")
        print(f"   Content: {received.content}")
    
    # Test 3: Multiple messages
    print("\n3️⃣ Testing Multiple Messages...\n")
    
    msg2 = create_data_message(
        from_agent="Researcher",
        to_agent="Writer",
        data={"research": "AI trends data..."},
        description="Research findings"
    )
    bus.send(msg2)
    
    msg3 = create_feedback_message(
        from_agent="Reviewer",
        to_agent="Writer",
        feedback="Please add more sources",
        suggestions=["Add 3 more citations", "Expand section 2"]
    )
    bus.send(msg3)
    
    # Test 4: Conversation manager
    print("\n4️⃣ Testing Conversation Manager...\n")
    
    conv_mgr = ConversationManager(bus)
    conv_id = conv_mgr.start_conversation(
        conversation_id="research_task_001",
        initiator="Orchestrator",
        recipient="Researcher",
        topic="AI Trends Research"
    )
    
    # Test 5: Statistics
    print("\n5️⃣ Message Bus Statistics...\n")
    
    stats = bus.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Test 6: Export history
    print("\n6️⃣ Exporting Message History...\n")
    
    os.makedirs("./outputs", exist_ok=True)
    bus.export_history("./outputs/message_history.json")
    
    print("\n" + "="*60)
    print("✅ Communication Protocol Tests Complete!")
    print("="*60 + "\n")