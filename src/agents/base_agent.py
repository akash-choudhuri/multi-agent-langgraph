"""
Base agent class for multi-agent system.
All specialized agents inherit from this base class.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..core.state import MultiAgentState, AgentOutput, AgentType, StateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for an agent."""
    agent_type: AgentType
    model_name: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the base agent."""
        self.config = config
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        try:
            if "gpt" in self.config.model_name.lower():
                return ChatOpenAI(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
            elif "claude" in self.config.model_name.lower():
                return ChatAnthropic(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    max_tokens_to_sample=self.config.max_tokens,
                    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            else:
                # Default fallback
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools for this agent. Override in subclasses."""
        return []
    
    @abstractmethod
    def get_system_message(self) -> str:
        """Get the system message for this agent."""
        pass
    
    @abstractmethod
    def process_task(self, state: MultiAgentState) -> MultiAgentState:
        """Process a task and update the state."""
        pass
    
    def _create_prompt(self, state: MultiAgentState, additional_context: str = "") -> str:
        """Create a prompt based on the current state."""
        prompt_parts = []
        
        # Add task context
        prompt_parts.append(f"Task Objective: {state['task_objective']}")
        prompt_parts.append(f"Original Query: {state['original_query']}")
        
        # Add requirements
        if state['task_requirements']:
            prompt_parts.append("\nTask Requirements:")
            for i, req in enumerate(state['task_requirements'], 1):
                status = "✅" if req.completed else "❌"
                prompt_parts.append(f"{i}. {req.description} {status}")
        
        # Add previous agent outputs
        if state['agent_outputs']:
            prompt_parts.append("\nPrevious Agent Outputs:")
            for agent_type, output in state['agent_outputs'].items():
                prompt_parts.append(f"\n{agent_type.upper()}:")
                prompt_parts.append(f"Content: {output.content}")
                prompt_parts.append(f"Confidence: {output.confidence}")
        
        # Add research data if available
        if state['research_data']:
            prompt_parts.append(f"\nAvailable Research Data: {len(state['research_data'])} items")
            for i, data in enumerate(state['research_data'][:3], 1):  # Show first 3
                prompt_parts.append(f"{i}. {data.get('title', 'Research Item')}: {data.get('summary', 'No summary')}")
        
        # Add analysis results if available
        if state['analysis_results']:
            prompt_parts.append(f"\nAnalysis Results Available: {list(state['analysis_results'].keys())}")
        
        # Add iteration context
        prompt_parts.append(f"\nIteration: {state['iteration_count']} / {state['max_iterations']}")
        
        # Add additional context
        if additional_context:
            prompt_parts.append(f"\nAdditional Context:\n{additional_context}")
        
        return "\n".join(prompt_parts)
    
    def _execute_llm_call(self, messages: List[Any]) -> str:
        """Execute LLM call with error handling."""
        try:
            start_time = time.time()
            response = self.llm.invoke(messages)
            execution_time = time.time() - start_time
            
            # Extract content based on response type
            if hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            logger.info(f"{self.config.agent_type} completed LLM call in {execution_time:.2f}s")
            return content
            
        except Exception as e:
            logger.error(f"LLM call failed for {self.config.agent_type}: {str(e)}")
            return f"Error processing request: {str(e)}"
    
    def _calculate_confidence(self, content: str, context: Dict[str, Any] = None) -> float:
        """Calculate confidence score for the output."""
        # Basic confidence calculation - can be enhanced
        base_confidence = 0.7
        
        # Increase confidence based on content length and structure
        if len(content) > 100:
            base_confidence += 0.1
        if len(content) > 500:
            base_confidence += 0.1
            
        # Decrease confidence if there are error indicators
        error_indicators = ["error", "failed", "unable", "cannot", "unclear"]
        if any(indicator in content.lower() for indicator in error_indicators):
            base_confidence -= 0.2
        
        # Ensure confidence is within valid range
        return max(0.0, min(1.0, base_confidence))
    
    def create_agent_output(
        self,
        content: str,
        sources: List[str] = None,
        metadata: Dict[str, Any] = None,
        execution_time: float = None
    ) -> AgentOutput:
        """Create standardized agent output."""
        return AgentOutput(
            agent_type=self.config.agent_type,
            content=content,
            confidence=self._calculate_confidence(content, metadata),
            sources=sources or [],
            metadata=metadata or {},
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def run(self, state: MultiAgentState) -> MultiAgentState:
        """Main execution method for the agent."""
        try:
            logger.info(f"Starting {self.config.agent_type} agent")
            start_time = time.time()
            
            # Update current agent in state
            state['current_agent'] = self.config.agent_type.value
            state = StateManager.set_task_status(state, "in_progress")
            
            # Process the task
            updated_state = self.process_task(state)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update execution time in the output if it exists
            if self.config.agent_type.value in updated_state['agent_outputs']:
                output = updated_state['agent_outputs'][self.config.agent_type.value]
                output.execution_time = execution_time
            
            logger.info(f"{self.config.agent_type} completed in {execution_time:.2f}s")
            return updated_state
            
        except Exception as e:
            logger.error(f"Error in {self.config.agent_type}: {str(e)}")
            state = StateManager.add_error(state, f"{self.config.agent_type}: {str(e)}")
            
            # Create error output
            error_output = self.create_agent_output(
                content=f"Agent encountered an error: {str(e)}",
                metadata={"error": True, "error_message": str(e)}
            )
            
            state = StateManager.update_agent_output(state, self.config.agent_type, error_output)
            return state
    
    def should_continue(self, state: MultiAgentState) -> bool:
        """Determine if the workflow should continue after this agent."""
        # Continue if:
        # 1. We haven't reached max iterations
        # 2. Task is not completed
        # 3. No critical errors
        
        if state['iteration_count'] >= state['max_iterations']:
            return False
            
        if state['task_status'] == "completed":
            return False
            
        if len(state['errors']) > 3:  # Too many errors
            return False
            
        return True
    
    def get_handoff_message(self, state: MultiAgentState, next_agent: str) -> str:
        """Generate a handoff message for the next agent."""
        current_output = state['agent_outputs'].get(self.config.agent_type.value)
        if not current_output:
            return f"Handoff from {self.config.agent_type} to {next_agent}."
            
        return f"Handoff from {self.config.agent_type} to {next_agent}. " \
               f"Completed work with confidence {current_output.confidence:.2f}. " \
               f"Key output: {current_output.content[:100]}..."