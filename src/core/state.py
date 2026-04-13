"""
Core state management for multi-agent system.
Defines the shared state structure used across all agents in the LangGraph workflow.
"""

from typing import Dict, List, Optional, Any, TypedDict
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


class AgentType(str, Enum):
    """Available agent types."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AgentOutput(BaseModel):
    """Standard output format for all agents."""
    agent_type: AgentType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskRequirement(BaseModel):
    """Individual task requirement specification."""
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    required_agents: List[AgentType] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    completed: bool = False


class MultiAgentState(TypedDict):
    """
    Shared state structure for the multi-agent system.
    This state is passed between all agents in the LangGraph workflow.
    """
    # Task Information
    task_id: str
    original_query: str
    task_objective: str
    task_requirements: List[TaskRequirement]
    task_status: TaskStatus
    priority: TaskPriority
    
    # Agent Outputs
    agent_outputs: Dict[str, AgentOutput]
    current_agent: Optional[str]
    next_agent: Optional[str]
    
    # Execution Context
    iteration_count: int
    max_iterations: int
    start_time: datetime
    last_updated: datetime
    
    # Shared Knowledge
    research_data: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    generated_content: List[str]
    quality_scores: Dict[str, float]
    
    # Collaboration Data
    agent_communications: List[Dict[str, Any]]
    handoff_notes: Dict[str, str]
    shared_context: Dict[str, Any]
    
    # Error Handling
    errors: List[str]
    warnings: List[str]
    retry_count: int
    
    # Final Results
    final_output: Optional[str]
    quality_assessment: Optional[Dict[str, Any]]
    execution_summary: Optional[Dict[str, Any]]


class StateManager:
    """Utility class for managing and updating the multi-agent state."""
    
    @staticmethod
    def create_initial_state(
        task_id: str,
        query: str,
        objective: str,
        requirements: List[TaskRequirement],
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_iterations: int = 3
    ) -> MultiAgentState:
        """Create initial state for a new task."""
        return MultiAgentState(
            # Task Information
            task_id=task_id,
            original_query=query,
            task_objective=objective,
            task_requirements=requirements,
            task_status=TaskStatus.PENDING,
            priority=priority,
            
            # Agent Outputs
            agent_outputs={},
            current_agent=None,
            next_agent=None,
            
            # Execution Context
            iteration_count=0,
            max_iterations=max_iterations,
            start_time=datetime.now(),
            last_updated=datetime.now(),
            
            # Shared Knowledge
            research_data=[],
            analysis_results={},
            generated_content=[],
            quality_scores={},
            
            # Collaboration Data
            agent_communications=[],
            handoff_notes={},
            shared_context={},
            
            # Error Handling
            errors=[],
            warnings=[],
            retry_count=0,
            
            # Final Results
            final_output=None,
            quality_assessment=None,
            execution_summary=None
        )
    
    @staticmethod
    def update_agent_output(
        state: MultiAgentState,
        agent_type: AgentType,
        output: AgentOutput
    ) -> MultiAgentState:
        """Update state with agent output."""
        state["agent_outputs"][agent_type.value] = output
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def add_research_data(
        state: MultiAgentState,
        data: Dict[str, Any]
    ) -> MultiAgentState:
        """Add research data to shared knowledge."""
        state["research_data"].append(data)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def update_analysis_results(
        state: MultiAgentState,
        results: Dict[str, Any]
    ) -> MultiAgentState:
        """Update analysis results."""
        state["analysis_results"].update(results)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def add_generated_content(
        state: MultiAgentState,
        content: str
    ) -> MultiAgentState:
        """Add generated content."""
        state["generated_content"].append(content)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def record_agent_communication(
        state: MultiAgentState,
        from_agent: str,
        to_agent: str,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> MultiAgentState:
        """Record communication between agents."""
        communication = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        state["agent_communications"].append(communication)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def add_error(
        state: MultiAgentState,
        error_message: str
    ) -> MultiAgentState:
        """Add error to state."""
        state["errors"].append(error_message)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def add_warning(
        state: MultiAgentState,
        warning_message: str
    ) -> MultiAgentState:
        """Add warning to state."""
        state["warnings"].append(warning_message)
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def increment_iteration(
        state: MultiAgentState
    ) -> MultiAgentState:
        """Increment iteration count."""
        state["iteration_count"] += 1
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def set_task_status(
        state: MultiAgentState,
        status: TaskStatus
    ) -> MultiAgentState:
        """Update task status."""
        state["task_status"] = status
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def set_final_output(
        state: MultiAgentState,
        output: str,
        quality_assessment: Dict[str, Any] = None
    ) -> MultiAgentState:
        """Set final output and complete task."""
        state["final_output"] = output
        state["quality_assessment"] = quality_assessment
        state["task_status"] = TaskStatus.COMPLETED
        state["last_updated"] = datetime.now()
        return state
    
    @staticmethod
    def get_execution_summary(state: MultiAgentState) -> Dict[str, Any]:
        """Generate execution summary from state."""
        total_time = (state["last_updated"] - state["start_time"]).total_seconds()
        
        return {
            "task_id": state["task_id"],
            "status": state["task_status"],
            "total_execution_time": total_time,
            "iterations": state["iteration_count"],
            "agents_involved": list(state["agent_outputs"].keys()),
            "research_items": len(state["research_data"]),
            "generated_content_items": len(state["generated_content"]),
            "communications": len(state["agent_communications"]),
            "errors": len(state["errors"]),
            "warnings": len(state["warnings"]),
            "quality_scores": state["quality_scores"],
            "final_output_length": len(state["final_output"]) if state["final_output"] else 0
        }