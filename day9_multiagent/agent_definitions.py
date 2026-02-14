"""
Agent Definitions - Day 9 Multi-Agent System
Defines agent roles, capabilities, and configurations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import json
from datetime import datetime


class AgentRole(Enum):
    """Predefined agent roles"""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    ORCHESTRATOR = "orchestrator"


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """
    Configuration for each agent
    
    Attributes:
        role: The role of the agent (RESEARCHER, WRITER, etc.)
        name: Display name for the agent
        goal: What the agent aims to achieve
        backstory: Context and expertise of the agent
        capabilities: List of what the agent can do
        max_iterations: Maximum task attempts
        temperature: LLM temperature (0.0-1.0)
        model: LLM model to use
    """
    role: AgentRole
    name: str
    goal: str
    backstory: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    max_iterations: int = 5
    temperature: float = 0.7
    model: str = "llama-3.3-70b-versatile"
    verbose: bool = True
    allow_delegation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "role": self.role.value,
            "name": self.name,
            "goal": self.goal,
            "backstory": self.backstory,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "model": self.model
        }


class BaseAgent:
    """
    Base Agent class that all agents inherit from
    
    This provides common functionality:
    - Message handling
    - Status tracking
    - Logging
    - Error handling
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.status = AgentStatus.IDLE
        self.message_history: List[Dict] = []
        self.task_results: List[Dict] = []
        self.errors: List[str] = []
        self.current_task = None
        
    def receive_message(self, message: Dict[str, Any]) -> None:
        """
        Receive a message from another agent
        
        Args:
            message: Message dictionary with 'from', 'content', 'type'
        """
        self.message_history.append({
            **message,
            "received_at": datetime.now().isoformat()
        })
        print(f"📨 {self.config.name} received message from {message.get('from', 'unknown')}")
        
    def send_message(self, to_agent: str, content: Any, message_type: str = "data") -> Dict:
        """
        Send a message to another agent
        
        Args:
            to_agent: Name of recipient agent
            content: Message content
            message_type: Type of message (data, request, feedback, etc.)
        
        Returns:
            Message dictionary
        """
        message = {
            "from": self.config.name,
            "to": to_agent,
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        print(f"📤 {self.config.name} → {to_agent}: {message_type}")
        return message
    
    def log_action(self, action: str, details: Any = None) -> None:
        """Log an action performed by the agent"""
        log_entry = {
            "agent": self.config.name,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "status": self.status.value
        }
        self.task_results.append(log_entry)
        
    def set_status(self, status: AgentStatus) -> None:
        """Update agent status"""
        self.status = status
        print(f"🔄 {self.config.name} status: {status.value.upper()}")
        
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle and log errors"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.errors.append({
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "task": self.current_task
        })
        print(f"❌ {self.config.name} error: {error_msg}")
        self.set_status(AgentStatus.FAILED)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get agent execution summary"""
        return {
            "agent": self.config.name,
            "role": self.config.role.value,
            "status": self.status.value,
            "messages_received": len(self.message_history),
            "tasks_completed": len([r for r in self.task_results if r.get("status") == "completed"]),
            "errors": len(self.errors),
            "error_details": self.errors
        }


# =============================================================================
# PREDEFINED AGENT CONFIGURATIONS
# =============================================================================

RESEARCHER_CONFIG = AgentConfig(
    role=AgentRole.RESEARCHER,
    name="Research Analyst",
    goal="Gather comprehensive, accurate, and up-to-date information from the internet",
    backstory="""You are a senior research analyst with 10 years of experience in 
    information gathering and data analysis. You excel at finding reliable sources,
    fact-checking information, and presenting research in a clear, organized manner.
    You have access to Google search and web scraping tools to gather the latest information.
    You always cite your sources and verify information from multiple sources.""",
    temperature=0.3,  # Lower for factual accuracy
    max_iterations=5,
    capabilities=[
        AgentCapability(
            name="google_search",
            description="Search Google for information on any topic"
        ),
        AgentCapability(
            name="web_scrape",
            description="Extract content from websites"
        ),
        AgentCapability(
            name="verify_sources",
            description="Cross-reference information from multiple sources"
        )
    ]
)

WRITER_CONFIG = AgentConfig(
    role=AgentRole.WRITER,
    name="Content Writer",
    goal="Create engaging, well-structured, and informative content based on research",
    backstory="""You are an expert content writer with 8 years of experience in 
    creating compelling articles, blog posts, and reports. You excel at transforming
    research data into engaging narratives. You understand SEO best practices, 
    proper content structure, and how to write for different audiences. You always
    cite sources and maintain factual accuracy while keeping content interesting.""",
    temperature=0.7,  # Higher for creativity
    max_iterations=3,
    capabilities=[
        AgentCapability(
            name="write_content",
            description="Create well-structured content from research"
        ),
        AgentCapability(
            name="format_document",
            description="Format content with proper headings, sections, citations"
        )
    ]
)

REVIEWER_CONFIG = AgentConfig(
    role=AgentRole.REVIEWER,
    name="Quality Reviewer",
    goal="Ensure content meets quality standards and is factually accurate",
    backstory="""You are a meticulous quality assurance specialist with 12 years 
    of experience in content review and editing. You have an eye for detail and 
    ensure that all content is accurate, well-written, properly formatted, and 
    cites sources correctly. You provide constructive feedback and suggest specific
    improvements. You check for grammar, style, factual accuracy, and coherence.""",
    temperature=0.2,  # Very low for consistency
    max_iterations=3,
    capabilities=[
        AgentCapability(
            name="quality_check",
            description="Review content for quality and accuracy"
        ),
        AgentCapability(
            name="provide_feedback",
            description="Give specific, actionable feedback"
        ),
        AgentCapability(
            name="verify_citations",
            description="Check if all sources are properly cited"
        )
    ]
)

ORCHESTRATOR_CONFIG = AgentConfig(
    role=AgentRole.ORCHESTRATOR,
    name="Workflow Orchestrator",
    goal="Coordinate agents to complete tasks efficiently and handle failures",
    backstory="""You are an experienced project manager who coordinates team members
    to deliver high-quality outputs. You break down complex tasks, assign them to
    the right agents, monitor progress, handle failures gracefully, and ensure
    the final deliverable meets all requirements.""",
    temperature=0.1,  # Very low for logical decisions
    max_iterations=10,
    allow_delegation=True,
    capabilities=[
        AgentCapability(
            name="task_planning",
            description="Break down user requests into agent tasks"
        ),
        AgentCapability(
            name="agent_coordination",
            description="Assign tasks and monitor progress"
        ),
        AgentCapability(
            name="error_handling",
            description="Handle agent failures and retry logic"
        )
    ]
)


def get_agent_config(role: AgentRole) -> AgentConfig:
    """
    Get predefined configuration for an agent role
    
    Args:
        role: The agent role
    
    Returns:
        AgentConfig for that role
    """
    configs = {
        AgentRole.RESEARCHER: RESEARCHER_CONFIG,
        AgentRole.WRITER: WRITER_CONFIG,
        AgentRole.REVIEWER: REVIEWER_CONFIG,
        AgentRole.ORCHESTRATOR: ORCHESTRATOR_CONFIG
    }
    return configs.get(role)


def print_agent_info(config: AgentConfig) -> None:
    """Print agent information in a formatted way"""
    print(f"\n{'='*60}")
    print(f"🤖 {config.name.upper()}")
    print(f"{'='*60}")
    print(f"Role: {config.role.value}")
    print(f"Goal: {config.goal}")
    print(f"Backstory: {config.backstory}")
    print(f"Capabilities: {', '.join([c.name for c in config.capabilities])}")
    print(f"Temperature: {config.temperature}")
    print(f"Max Iterations: {config.max_iterations}")
    print(f"{'='*60}\n")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n🎭 AGENT DEFINITIONS - DAY 9 MULTI-AGENT SYSTEM\n")
    
    # Show all agent configurations
    for role in AgentRole:
        if role != AgentRole.ORCHESTRATOR:  # Skip orchestrator for now
            config = get_agent_config(role)
            if config:
                print_agent_info(config)
    
    # Test base agent functionality
    print("\n📝 Testing Base Agent Functionality...\n")
    
    test_agent = BaseAgent(RESEARCHER_CONFIG)
    
    # Test status changes
    test_agent.set_status(AgentStatus.WORKING)
    
    # Test message handling
    message = test_agent.send_message(
        to_agent="Writer",
        content="Here is the research data",
        message_type="data"
    )
    print(f"\nMessage sent: {json.dumps(message, indent=2)}")
    
    # Test action logging
    test_agent.log_action("google_search", {"query": "AI trends 2025"})
    
    # Get summary
    summary = test_agent.get_summary()
    print(f"\nAgent Summary:\n{json.dumps(summary, indent=2)}")
