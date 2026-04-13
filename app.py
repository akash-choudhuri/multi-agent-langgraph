"""
Streamlit Multi-Agent AI Application.
Frontend interface for the multi-agent system using LangGraph.
"""

import streamlit as st
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.workflows.multi_agent_workflow import MultiAgentWorkflow
from src.core.state import TaskPriority

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic AI - Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .agent-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-completed {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-running {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f1b0b7;
    }
    
    .quality-score {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .quality-high { color: #28a745; }
    .quality-medium { color: #ffc107; }
    .quality-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_workflow():
    """Initialize the multi-agent workflow system."""
    try:
        workflow = MultiAgentWorkflow()
        return workflow
    except Exception as e:
        st.error(f"Failed to initialize workflow: {str(e)}")
        return None


def display_agent_status(agent_outputs, current_agent=None):
    """Display current status of all agents."""
    st.subheader("🤖 Agent Status")
    
    agents = ['researcher', 'analyst', 'writer']
    
    cols = st.columns(len(agents))
    
    for i, agent in enumerate(agents):
        with cols[i]:
            # Agent card
            if agent in agent_outputs:
                output = agent_outputs[agent]
                confidence = output.get('confidence', 0)
                
                # Status badge
                if current_agent == agent:
                    status_class = "status-running"
                    status_text = "🔄 Running"
                else:
                    status_class = "status-completed"
                    status_text = "✅ Completed"
                
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent.title()} Agent</h4>
                    <div class="status-badge {status_class}">{status_text}</div>
                    <p><strong>Confidence:</strong> {confidence:.2f}</p>
                    <p><strong>Sources:</strong> {len(output.get('sources', []))}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="agent-card">
                    <h4>{agent.title()} Agent</h4>
                    <div class="status-badge">⏳ Pending</div>
                </div>
                """, unsafe_allow_html=True)


def display_quality_assessment(quality_assessment):
    """Display quality assessment results."""
    if not quality_assessment:
        return
    
    st.subheader("📊 Quality Assessment")
    
    overall_score = quality_assessment.get('overall_score', 0)
    individual_scores = quality_assessment.get('individual_scores', {})
    
    # Overall score
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if overall_score >= 0.8:
            quality_class = "quality-high"
            quality_label = "High Quality"
        elif overall_score >= 0.6:
            quality_class = "quality-medium"
            quality_label = "Medium Quality"
        else:
            quality_class = "quality-low"
            quality_label = "Needs Improvement"
        
        st.markdown(f"""
        <div class="quality-score {quality_class}">
            {overall_score:.2f}
        </div>
        <p><strong>{quality_label}</strong></p>
        """, unsafe_allow_html=True)
    
    with col2:
        # Individual scores
        for component, score in individual_scores.items():
            progress_color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
            st.write(f"**{component.title()}:** {score:.2f}")
            st.progress(score)
    
    # Recommendations
    recommendations = quality_assessment.get('recommendations', [])
    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")


def display_execution_summary(execution_summary):
    """Display execution summary."""
    if not execution_summary:
        return
    
    st.subheader("📈 Execution Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Time", f"{execution_summary.get('total_execution_time', 0):.1f}s")
    
    with col2:
        st.metric("Iterations", execution_summary.get('iterations', 0))
    
    with col3:
        st.metric("Agents Used", len(execution_summary.get('agents_involved', [])))
    
    with col4:
        st.metric("Research Items", execution_summary.get('research_items', 0))


async def run_workflow(workflow, query, objective, requirements, priority, max_iterations):
    """Run the multi-agent workflow."""
    try:
        result = await workflow.execute(
            query=query,
            objective=objective,
            requirements=requirements,
            priority=TaskPriority(priority.lower()),
            max_iterations=max_iterations
        )
        return result
    except Exception as e:
        st.error(f"Workflow execution failed: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Agentic AI</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Agent System with LangGraph for Complex Task Orchestration**")
    
    # Initialize workflow
    with st.spinner("Initializing multi-agent system..."):
        workflow = initialize_workflow()
    
    if not workflow:
        st.error("Failed to initialize the multi-agent system. Please check your configuration.")
        return
    
    # Sidebar configuration
    st.sidebar.title("🔧 Configuration")
    
    # Workflow info
    workflow_info = workflow.get_workflow_info()
    
    with st.sidebar.expander("ℹ️ System Information"):
        st.write(f"**Agents:** {', '.join(workflow_info['agents'])}")
        st.write(f"**Workflow Nodes:** {len(workflow_info['workflow_nodes'])}")
        
        for agent, model in workflow_info['agent_models'].items():
            st.write(f"**{agent.title()}:** {model}")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🚀 Task Execution", "📊 Results Dashboard", "📋 Task History"])
    
    # Task Execution Tab
    with tab1:
        st.header("Task Execution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Task input
            st.subheader("📝 Task Definition")
            
            query = st.text_area(
                "Query/Question",
                placeholder="Enter your question or task description...",
                height=100,
                help="Describe what you want the multi-agent system to research and analyze"
            )
            
            objective = st.text_input(
                "Objective (Optional)",
                placeholder="Specific objective or goal...",
                help="If different from the query, specify the main objective"
            )
            
            # Requirements
            st.subheader("📋 Requirements")
            requirements_text = st.text_area(
                "Task Requirements (one per line)",
                placeholder="Requirement 1\nRequirement 2\nRequirement 3",
                height=100,
                help="Enter specific requirements, one per line"
            )
            
            requirements = [req.strip() for req in requirements_text.split('\n') if req.strip()] if requirements_text else []
        
        with col2:
            # Execution parameters
            st.subheader("⚙️ Parameters")
            
            priority = st.selectbox(
                "Priority",
                options=["low", "medium", "high", "urgent"],
                index=1,
                help="Task priority level"
            )
            
            max_iterations = st.slider(
                "Max Iterations",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum number of refinement iterations"
            )
            
            # Execute button
            if st.button("🚀 Execute Task", type="primary", disabled=not query):
                if not query:
                    st.error("Please enter a query first!")
                else:
                    # Store task in session state
                    st.session_state.current_task = {
                        'query': query,
                        'objective': objective or query,
                        'requirements': requirements,
                        'priority': priority,
                        'max_iterations': max_iterations,
                        'timestamp': datetime.now()
                    }
                    
                    # Execute workflow
                    with st.spinner("🤖 Multi-agent system is working..."):
                        # Create placeholder for real-time updates
                        status_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        
                        # Simulate progress updates
                        for i, agent in enumerate(['researcher', 'analyst', 'writer']):
                            status_placeholder.info(f"🔄 {agent.title()} Agent is working...")
                            progress_bar.progress((i + 1) / 4)
                            
                        # Execute the workflow
                        result = asyncio.run(run_workflow(
                            workflow, query, objective or query, requirements, priority, max_iterations
                        ))
                        
                        progress_bar.progress(1.0)
                        status_placeholder.success("✅ Task completed!")
                        
                        if result:
                            st.session_state.last_result = result
                            st.rerun()
        
        # Display current task status
        if hasattr(st.session_state, 'current_task'):
            st.subheader("📊 Current Task")
            task = st.session_state.current_task
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Query:** {task['query'][:100]}...")
            with col2:
                st.write(f"**Priority:** {task['priority'].title()}")
            with col3:
                st.write(f"**Started:** {task['timestamp'].strftime('%H:%M:%S')}")
    
    # Results Dashboard Tab
    with tab2:
        st.header("Results Dashboard")
        
        if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
            result = st.session_state.last_result
            
            # Task status
            status = result.get('status', 'unknown')
            if status == 'completed':
                st.success(f"✅ Task completed successfully! (ID: {result.get('task_id', 'N/A')})")
            elif status == 'failed':
                st.error(f"❌ Task failed. Error: {result.get('error', 'Unknown error')}")
            else:
                st.info(f"ℹ️ Task status: {status}")
            
            # Agent outputs
            agent_outputs = result.get('agent_outputs', {})
            if agent_outputs:
                display_agent_status(agent_outputs)
            
            # Quality assessment
            quality_assessment = result.get('quality_assessment')
            if quality_assessment:
                display_quality_assessment(quality_assessment)
            
            # Execution summary
            execution_summary = result.get('execution_summary')
            if execution_summary:
                display_execution_summary(execution_summary)
            
            # Final output
            final_output = result.get('final_output')
            if final_output:
                st.subheader("📄 Final Output")
                
                # Download button
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Result",
                        data=final_output,
                        file_name=f"agentic_ai_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                # Display content
                st.markdown(final_output)
            
            # Errors and warnings
            errors = result.get('errors', [])
            warnings = result.get('warnings', [])
            
            if errors:
                st.subheader("❌ Errors")
                for error in errors:
                    st.error(error)
            
            if warnings:
                st.subheader("⚠️ Warnings")
                for warning in warnings:
                    st.warning(warning)
            
            # Individual agent outputs
            if agent_outputs:
                st.subheader("🤖 Detailed Agent Outputs")
                
                for agent_name, output in agent_outputs.items():
                    with st.expander(f"{agent_name.title()} Agent Output"):
                        st.write(f"**Confidence:** {output.get('confidence', 0):.2f}")
                        st.write(f"**Sources:** {len(output.get('sources', []))}")
                        
                        metadata = output.get('metadata', {})
                        if metadata:
                            st.write("**Metadata:**")
                            st.json(metadata)
                        
                        st.write("**Content:**")
                        st.text_area("", output.get('content', ''), height=200, key=f"content_{agent_name}")
        
        else:
            st.info("🚀 Execute a task to see results here")
    
    # Task History Tab
    with tab3:
        st.header("Task History")
        
        # Initialize task history in session state
        if 'task_history' not in st.session_state:
            st.session_state.task_history = []
        
        # Add completed task to history
        if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
            result = st.session_state.last_result
            task = getattr(st.session_state, 'current_task', {})
            
            # Check if this result is already in history
            if not any(h.get('task_id') == result.get('task_id') for h in st.session_state.task_history):
                history_item = {
                    'task_id': result.get('task_id'),
                    'query': task.get('query', 'Unknown'),
                    'status': result.get('status'),
                    'timestamp': task.get('timestamp', datetime.now()),
                    'execution_time': result.get('execution_summary', {}).get('total_execution_time', 0),
                    'quality_score': result.get('quality_assessment', {}).get('overall_score', 0)
                }
                st.session_state.task_history.append(history_item)
        
        # Display task history
        if st.session_state.task_history:
            st.write(f"**Total tasks executed:** {len(st.session_state.task_history)}")
            
            for i, task in enumerate(reversed(st.session_state.task_history)):
                with st.expander(f"Task {len(st.session_state.task_history) - i}: {task['query'][:50]}..."):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Status:** {task['status']}")
                    with col2:
                        st.write(f"**Duration:** {task['execution_time']:.1f}s")
                    with col3:
                        st.write(f"**Quality:** {task['quality_score']:.2f}")
                    with col4:
                        st.write(f"**Time:** {task['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("📝 No tasks executed yet")
        
        # Clear history button
        if st.session_state.task_history:
            if st.button("🗑️ Clear History"):
                st.session_state.task_history = []
                st.rerun()


if __name__ == "__main__":
    main()