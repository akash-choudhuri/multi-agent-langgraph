"""
Multi-Agent Workflow using LangGraph.
Orchestrates collaboration between different specialized agents.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from ..core.state import MultiAgentState, StateManager, TaskRequirement, TaskPriority, TaskStatus, AgentType
from ..agents.base_agent import AgentConfig
from ..agents.research_agent import ResearchAgent
from ..agents.analyst_agent import AnalystAgent  
from ..agents.writer_agent import WriterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """LangGraph-based multi-agent workflow orchestrator."""
    
    def __init__(self):
        """Initialize the multi-agent workflow."""
        self.agents = self._initialize_agents()
        self.workflow = self._build_workflow()
        
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents with their configurations."""
        agents = {}
        
        # Research Agent
        research_config = AgentConfig(
            agent_type=AgentType.RESEARCHER,
            model_name="gpt-4",
            temperature=0.1,
            max_tokens=3000
        )
        agents['researcher'] = ResearchAgent(research_config)
        
        # Analyst Agent
        analyst_config = AgentConfig(
            agent_type=AgentType.ANALYST,
            model_name="gpt-4",
            temperature=0.2,
            max_tokens=3000
        )
        agents['analyst'] = AnalystAgent(analyst_config)
        
        # Writer Agent
        writer_config = AgentConfig(
            agent_type=AgentType.WRITER,
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=4000
        )
        agents['writer'] = WriterAgent(writer_config)
        
        return agents
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes for each agent
        workflow.add_node("researcher", self._research_node)
        workflow.add_node("analyst", self._analyst_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("reviewer", self._reviewer_node)
        workflow.add_node("finalizer", self._finalizer_node)
        
        # Define the workflow edges
        workflow.set_entry_point("researcher")
        
        # Research -> Analysis
        workflow.add_edge("researcher", "analyst")
        
        # Analysis -> Writing
        workflow.add_edge("analyst", "writer")
        
        # Writing -> Review
        workflow.add_edge("writer", "reviewer")
        
        # Review -> End or iterate
        workflow.add_conditional_edges(
            "reviewer",
            self._should_continue,
            {
                "continue": "researcher",  # Iterate for improvement
                "finalize": "finalizer",   # Move to finalization
                "end": END                 # Complete the workflow
            }
        )
        
        # Finalizer -> End
        workflow.add_edge("finalizer", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _research_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute research agent."""
        logger.info("🔍 Executing Research Agent")
        try:
            updated_state = self.agents['researcher'].run(state)
            updated_state['current_agent'] = 'researcher'
            return updated_state
        except Exception as e:
            logger.error(f"Research agent error: {str(e)}")
            state = StateManager.add_error(state, f"Research agent failed: {str(e)}")
            return state
    
    def _analyst_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute analyst agent."""
        logger.info("📊 Executing Analyst Agent")
        try:
            updated_state = self.agents['analyst'].run(state)
            updated_state['current_agent'] = 'analyst'
            return updated_state
        except Exception as e:
            logger.error(f"Analyst agent error: {str(e)}")
            state = StateManager.add_error(state, f"Analyst agent failed: {str(e)}")
            return state
    
    def _writer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute writer agent."""
        logger.info("✍️ Executing Writer Agent")
        try:
            updated_state = self.agents['writer'].run(state)
            updated_state['current_agent'] = 'writer'
            return updated_state
        except Exception as e:
            logger.error(f"Writer agent error: {str(e)}")
            state = StateManager.add_error(state, f"Writer agent failed: {str(e)}")
            return state
    
    def _reviewer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Execute quality review process."""
        logger.info("🔍 Executing Quality Review")
        
        try:
            # Perform quality assessment
            quality_assessment = self._assess_quality(state)
            
            # Update state with quality assessment
            state['quality_assessment'] = quality_assessment
            state['current_agent'] = 'reviewer'
            
            # Determine if quality is acceptable
            overall_score = quality_assessment.get('overall_score', 0.0)
            if overall_score >= 0.8:
                state = StateManager.set_task_status(state, TaskStatus.COMPLETED)
            elif state['iteration_count'] >= state['max_iterations']:
                state = StateManager.set_task_status(state, TaskStatus.COMPLETED)
                state = StateManager.add_warning(state, "Max iterations reached, finalizing with current quality")
            else:
                state = StateManager.set_task_status(state, TaskStatus.REQUIRES_REVIEW)
            
            logger.info(f"Quality assessment complete. Overall score: {overall_score:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Review process error: {str(e)}")
            state = StateManager.add_error(state, f"Review process failed: {str(e)}")
            return state
    
    def _finalizer_node(self, state: MultiAgentState) -> MultiAgentState:
        """Finalize the workflow and prepare output."""
        logger.info("✨ Finalizing Workflow")
        
        try:
            # Get the final content from writer agent
            writer_output = state['agent_outputs'].get('writer')
            if writer_output:
                final_content = writer_output.content
            else:
                final_content = "No final content generated"
            
            # Set final output
            state = StateManager.set_final_output(
                state, 
                final_content, 
                state.get('quality_assessment', {})
            )
            
            # Generate execution summary
            execution_summary = StateManager.get_execution_summary(state)
            state['execution_summary'] = execution_summary
            
            logger.info("Workflow finalized successfully")
            return state
            
        except Exception as e:
            logger.error(f"Finalization error: {str(e)}")
            state = StateManager.add_error(state, f"Finalization failed: {str(e)}")
            return state
    
    def _should_continue(self, state: MultiAgentState) -> str:
        """Determine next step in the workflow."""
        # Check if task is completed
        if state['task_status'] == TaskStatus.COMPLETED:
            # If we have good quality, finalize
            quality_score = state.get('quality_assessment', {}).get('overall_score', 0.0)
            if quality_score >= 0.8 or state['iteration_count'] >= state['max_iterations']:
                return "finalize"
            return "end"
        
        # Check if we should continue iterating
        if state['iteration_count'] < state['max_iterations'] and state['task_status'] == TaskStatus.REQUIRES_REVIEW:
            # Increment iteration count
            state = StateManager.increment_iteration(state)
            return "continue"
        
        # Default to finalization
        return "finalize"
    
    def _assess_quality(self, state: MultiAgentState) -> Dict[str, Any]:
        """Assess the quality of the current workflow output."""
        quality_scores = {}
        
        # Check research quality
        research_output = state['agent_outputs'].get('researcher')
        if research_output:
            research_quality = self._assess_research_quality(research_output, state)
            quality_scores['research'] = research_quality
        
        # Check analysis quality  
        analyst_output = state['agent_outputs'].get('analyst')
        if analyst_output:
            analysis_quality = self._assess_analysis_quality(analyst_output, state)
            quality_scores['analysis'] = analysis_quality
        
        # Check writing quality
        writer_output = state['agent_outputs'].get('writer')
        if writer_output:
            writing_quality = self._assess_writing_quality(writer_output, state)
            quality_scores['writing'] = writing_quality
        
        # Calculate overall quality
        if quality_scores:
            overall_score = sum(quality_scores.values()) / len(quality_scores)
        else:
            overall_score = 0.0
        
        return {
            'individual_scores': quality_scores,
            'overall_score': overall_score,
            'assessment_timestamp': datetime.now().isoformat(),
            'criteria_met': overall_score >= 0.8,
            'recommendations': self._generate_quality_recommendations(quality_scores, state)
        }
    
    def _assess_research_quality(self, research_output, state: MultiAgentState) -> float:
        """Assess research quality."""
        score = 0.5  # Base score
        
        # Check confidence level
        if research_output.confidence > 0.8:
            score += 0.2
        elif research_output.confidence > 0.6:
            score += 0.1
        
        # Check number of sources
        if len(research_output.sources) >= 3:
            score += 0.2
        elif len(research_output.sources) >= 1:
            score += 0.1
        
        # Check content length
        if len(research_output.content) > 1000:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_analysis_quality(self, analyst_output, state: MultiAgentState) -> float:
        """Assess analysis quality."""
        score = 0.5  # Base score
        
        # Check confidence level
        if analyst_output.confidence > 0.8:
            score += 0.2
        elif analyst_output.confidence > 0.6:
            score += 0.1
        
        # Check if insights were generated
        insights = state['analysis_results'].get('insights', [])
        if len(insights) >= 3:
            score += 0.2
        elif len(insights) >= 1:
            score += 0.1
        
        # Check data utilization
        if len(state['research_data']) > 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_writing_quality(self, writer_output, state: MultiAgentState) -> float:
        """Assess writing quality."""
        score = 0.5  # Base score
        
        # Check confidence level
        if writer_output.confidence > 0.8:
            score += 0.2
        elif writer_output.confidence > 0.6:
            score += 0.1
        
        # Check content structure
        metadata = writer_output.metadata
        if metadata.get('sections_count', 0) >= 3:
            score += 0.1
        
        if metadata.get('includes_summary', False):
            score += 0.1
        
        if metadata.get('word_count', 0) > 500:
            score += 0.1
        
        return min(1.0, score)
    
    def _generate_quality_recommendations(self, quality_scores: Dict[str, float], state: MultiAgentState) -> List[str]:
        """Generate recommendations for quality improvement."""
        recommendations = []
        
        for component, score in quality_scores.items():
            if score < 0.7:
                if component == 'research':
                    recommendations.append("Research component needs more comprehensive source gathering")
                elif component == 'analysis':
                    recommendations.append("Analysis component needs deeper insight generation")
                elif component == 'writing':
                    recommendations.append("Writing component needs better structure and clarity")
        
        if len(state['errors']) > 0:
            recommendations.append("Address system errors that occurred during processing")
        
        return recommendations
    
    async def execute(
        self,
        query: str,
        objective: str = None,
        requirements: List[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Execute the multi-agent workflow."""
        try:
            # Create task ID
            task_id = str(uuid.uuid4())
            
            # Set objective if not provided
            if objective is None:
                objective = query
            
            # Create task requirements
            task_requirements = []
            if requirements:
                for req_desc in requirements:
                    task_requirements.append(
                        TaskRequirement(
                            description=req_desc,
                            priority=priority
                        )
                    )
            else:
                # Default requirements
                task_requirements = [
                    TaskRequirement(description="Conduct thorough research on the topic"),
                    TaskRequirement(description="Analyze findings and generate insights"),
                    TaskRequirement(description="Create comprehensive written content")
                ]
            
            # Create initial state
            initial_state = StateManager.create_initial_state(
                task_id=task_id,
                query=query,
                objective=objective,
                requirements=task_requirements,
                priority=priority,
                max_iterations=max_iterations
            )
            
            logger.info(f"🚀 Starting multi-agent workflow for task: {task_id}")
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Extract results
            result = {
                'task_id': task_id,
                'status': final_state['task_status'],
                'final_output': final_state.get('final_output'),
                'quality_assessment': final_state.get('quality_assessment'),
                'execution_summary': final_state.get('execution_summary'),
                'agent_outputs': {
                    agent_type: {
                        'content': output.content,
                        'confidence': output.confidence,
                        'sources': output.sources,
                        'metadata': output.metadata
                    }
                    for agent_type, output in final_state['agent_outputs'].items()
                },
                'errors': final_state['errors'],
                'warnings': final_state['warnings']
            }
            
            logger.info(f"✅ Workflow completed successfully for task: {task_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {str(e)}")
            return {
                'task_id': task_id if 'task_id' in locals() else 'unknown',
                'status': 'failed',
                'error': str(e),
                'final_output': None
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow configuration."""
        return {
            'agents': list(self.agents.keys()),
            'workflow_nodes': ['researcher', 'analyst', 'writer', 'reviewer', 'finalizer'],
            'max_iterations_default': 3,
            'supported_priorities': [p.value for p in TaskPriority],
            'agent_models': {
                name: agent.config.model_name 
                for name, agent in self.agents.items()
            }
        }