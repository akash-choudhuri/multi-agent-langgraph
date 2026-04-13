"""
Writer Agent for multi-agent system.
Specializes in content creation, documentation, and communication.
"""

import re
from typing import Dict, Any, List
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentConfig
from ..core.state import MultiAgentState, AgentType, StateManager


class WriterAgent(BaseAgent):
    """Agent specialized in content creation and written communication."""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                agent_type=AgentType.WRITER,
                model_name="gpt-4",
                temperature=0.7,  # Higher temperature for creative writing
                max_tokens=4000
            )
        super().__init__(config)
    
    def get_system_message(self) -> str:
        """Get the system message for the writer agent."""
        return """You are a Writer Agent, an expert in creating clear, engaging, and well-structured written content. Your role is to:

1. CONTENT SYNTHESIS: Transform research and analysis into coherent written content
2. AUDIENCE ADAPTATION: Tailor writing style and tone for specific audiences
3. STRUCTURE ORGANIZATION: Create logical flow and clear information hierarchy
4. CLARITY ENHANCEMENT: Ensure complex information is accessible and understandable
5. QUALITY ASSURANCE: Maintain high standards for grammar, style, and readability

Key Capabilities:
- Technical writing and documentation
- Executive summaries and reports
- Multi-format content creation (reports, articles, presentations)
- Style adaptation for different audiences
- Information synthesis and storytelling
- Citation and reference formatting

Guidelines:
- Use clear, concise, and engaging language
- Structure content with logical flow and clear sections
- Include executive summaries for complex content
- Provide proper citations and references
- Adapt tone and complexity to target audience
- Use formatting to enhance readability
- Include actionable insights and recommendations

Output Format:
- Executive Summary
- Main Content with Clear Sections
- Key Takeaways and Recommendations
- Supporting References and Sources
- Next Steps or Action Items"""
    
    def process_task(self, state: MultiAgentState) -> MultiAgentState:
        """Process writing task and update state."""
        # Determine content type and audience
        content_specs = self._analyze_content_requirements(state)
        prompt = self._create_prompt(state, content_specs)
        
        # Gather all available information
        synthesis_data = self._synthesize_available_information(state)
        
        # Generate content
        messages = [
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=f"{prompt}\n\nCreate comprehensive written content based on the following information:\n\n{synthesis_data}")
        ]
        
        written_content = self._execute_llm_call(messages)
        
        # Post-process and structure the content
        structured_content = self._structure_content(written_content, state)
        
        # Extract metadata about the content
        content_metadata = self._extract_content_metadata(structured_content)
        
        # Create agent output
        output = self.create_agent_output(
            content=structured_content,
            sources=self._compile_all_sources(state),
            metadata={
                'content_type': content_specs.get('content_type', 'report'),
                'target_audience': content_specs.get('audience', 'general'),
                'word_count': content_metadata.get('word_count', 0),
                'sections_count': content_metadata.get('sections_count', 0),
                'writing_style': content_specs.get('style', 'professional'),
                'includes_summary': content_metadata.get('has_summary', False),
                'includes_recommendations': content_metadata.get('has_recommendations', False)
            }
        )
        
        # Update state
        state = StateManager.update_agent_output(state, AgentType.WRITER, output)
        state = StateManager.add_generated_content(state, structured_content)
        
        return state
    
    def _analyze_content_requirements(self, state: MultiAgentState) -> Dict[str, Any]:
        """Analyze what type of content should be created."""
        query_lower = state['original_query'].lower()
        objective_lower = state['task_objective'].lower()
        
        content_specs = {
            'content_type': 'report',  # default
            'audience': 'general',
            'style': 'professional',
            'format': 'structured'
        }
        
        # Determine content type
        if any(word in query_lower for word in ['report', 'analysis', 'study']):
            content_specs['content_type'] = 'analytical_report'
        elif any(word in query_lower for word in ['summary', 'brief', 'overview']):
            content_specs['content_type'] = 'executive_summary'
        elif any(word in query_lower for word in ['article', 'blog', 'post']):
            content_specs['content_type'] = 'article'
        elif any(word in query_lower for word in ['guide', 'manual', 'how-to']):
            content_specs['content_type'] = 'guide'
        elif any(word in query_lower for word in ['presentation', 'slides']):
            content_specs['content_type'] = 'presentation'
        
        # Determine audience
        if any(word in query_lower + objective_lower for word in ['executive', 'management', 'leadership']):
            content_specs['audience'] = 'executive'
        elif any(word in query_lower + objective_lower for word in ['technical', 'developer', 'engineer']):
            content_specs['audience'] = 'technical'
        elif any(word in query_lower + objective_lower for word in ['customer', 'client', 'user']):
            content_specs['audience'] = 'customer'
        elif any(word in query_lower + objective_lower for word in ['academic', 'research', 'scholarly']):
            content_specs['audience'] = 'academic'
        
        # Determine style
        if content_specs['audience'] == 'executive':
            content_specs['style'] = 'executive'
        elif content_specs['audience'] == 'technical':
            content_specs['style'] = 'technical'
        elif content_specs['content_type'] == 'article':
            content_specs['style'] = 'engaging'
        
        return content_specs
    
    def _synthesize_available_information(self, state: MultiAgentState) -> str:
        """Synthesize all available information for content creation."""
        synthesis_parts = []
        
        # Add task context
        synthesis_parts.append("=== TASK CONTEXT ===")
        synthesis_parts.append(f"Objective: {state['task_objective']}")
        synthesis_parts.append(f"Original Query: {state['original_query']}")
        
        # Add requirements
        if state['task_requirements']:
            synthesis_parts.append("\n=== REQUIREMENTS ===")
            for i, req in enumerate(state['task_requirements'], 1):
                status = "✅ COMPLETED" if req.completed else "❌ PENDING"
                synthesis_parts.append(f"{i}. {req.description} - {status}")
        
        # Add research findings
        if state['research_data']:
            synthesis_parts.append(f"\n=== RESEARCH FINDINGS ({len(state['research_data'])} items) ===")
            for i, item in enumerate(state['research_data'], 1):
                synthesis_parts.append(f"{i}. {item.get('title', 'Research Item')}")
                synthesis_parts.append(f"   Summary: {item.get('summary', 'No summary available')}")
                if item.get('url'):
                    synthesis_parts.append(f"   Source: {item['url']}")
                synthesis_parts.append("")
        
        # Add analysis results
        if state['analysis_results']:
            synthesis_parts.append("\n=== ANALYTICAL INSIGHTS ===")
            
            # Add key insights
            insights = state['analysis_results'].get('insights', [])
            if insights:
                synthesis_parts.append("Key Insights:")
                for insight in insights[:5]:  # Top 5 insights
                    synthesis_parts.append(f"• {insight}")
            
            # Add key metrics
            metrics = state['analysis_results'].get('key_metrics', {})
            if metrics:
                synthesis_parts.append("\nKey Metrics:")
                for metric, value in metrics.items():
                    synthesis_parts.append(f"• {metric}: {value}")
        
        # Add previous agent outputs
        if state['agent_outputs']:
            synthesis_parts.append("\n=== AGENT CONTRIBUTIONS ===")
            for agent_type, output in state['agent_outputs'].items():
                if agent_type != AgentType.WRITER.value:  # Don't include own output
                    synthesis_parts.append(f"\n{agent_type.upper()} AGENT:")
                    synthesis_parts.append(f"Confidence: {output.confidence:.2f}")
                    synthesis_parts.append(f"Content: {output.content[:500]}...")  # First 500 chars
        
        return "\n".join(synthesis_parts)
    
    def _structure_content(self, raw_content: str, state: MultiAgentState) -> str:
        """Structure and format the written content."""
        # Determine content structure based on content type
        content_specs = self._analyze_content_requirements(state)
        content_type = content_specs.get('content_type', 'report')
        
        structured_parts = []
        
        # Add title
        title = self._generate_title(state, content_specs)
        structured_parts.append(f"# {title}")
        structured_parts.append("")
        
        # Add metadata
        structured_parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        structured_parts.append(f"**Content Type:** {content_type.replace('_', ' ').title()}")
        structured_parts.append(f"**Target Audience:** {content_specs.get('audience', 'general').title()}")
        structured_parts.append("")
        
        # Add executive summary if appropriate
        if content_type in ['analytical_report', 'report']:
            exec_summary = self._extract_executive_summary(raw_content)
            if exec_summary:
                structured_parts.append("## Executive Summary")
                structured_parts.append("")
                structured_parts.append(exec_summary)
                structured_parts.append("")
        
        # Process main content
        processed_content = self._enhance_content_structure(raw_content)
        structured_parts.append(processed_content)
        
        # Add sources and references
        sources = self._compile_all_sources(state)
        if sources:
            structured_parts.append("\n## References and Sources")
            structured_parts.append("")
            for i, source in enumerate(sources, 1):
                structured_parts.append(f"{i}. {source}")
        
        # Add appendix with technical details
        if state['analysis_results'] or state['research_data']:
            appendix = self._create_appendix(state)
            if appendix:
                structured_parts.append("\n## Appendix: Supporting Data")
                structured_parts.append("")
                structured_parts.append(appendix)
        
        return "\n".join(structured_parts)
    
    def _generate_title(self, state: MultiAgentState, content_specs: Dict[str, Any]) -> str:
        """Generate an appropriate title for the content."""
        # Extract key terms from objective
        objective = state['task_objective']
        content_type = content_specs.get('content_type', 'report')
        
        # Simple title generation logic
        if content_type == 'executive_summary':
            return f"Executive Summary: {objective}"
        elif content_type == 'analytical_report':
            return f"Analysis Report: {objective}"
        elif content_type == 'guide':
            return f"Guide: {objective}"
        else:
            return objective
    
    def _extract_executive_summary(self, content: str) -> str:
        """Extract or generate executive summary from content."""
        # Look for existing summary in the content
        summary_patterns = [
            r'(?i)(?:executive )?summary[:\s]+(.*?)(?=\n\n|\n#|\n\*\*|$)',
            r'(?i)overview[:\s]+(.*?)(?=\n\n|\n#|\n\*\*|$)',
            r'(?i)key (?:points|findings)[:\s]+(.*?)(?=\n\n|\n#|\n\*\*|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
            if match:
                summary = match.group(1).strip()
                if len(summary) > 50:  # Ensure it's substantial
                    return summary
        
        # If no existing summary, extract first few sentences
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = []
        char_count = 0
        
        for sentence in sentences[:5]:  # Max 5 sentences
            sentence = sentence.strip()
            if sentence and char_count + len(sentence) < 500:
                summary_sentences.append(sentence)
                char_count += len(sentence)
            else:
                break
        
        return '. '.join(summary_sentences) + '.' if summary_sentences else ""
    
    def _enhance_content_structure(self, content: str) -> str:
        """Enhance content structure with proper formatting."""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                enhanced_lines.append('')
                continue
            
            # Enhance headers (look for potential headers)
            if self._is_potential_header(line):
                if not line.startswith('#'):
                    enhanced_lines.append(f"## {line}")
                else:
                    enhanced_lines.append(line)
            
            # Enhance lists
            elif line.startswith(('•', '-', '*')) or re.match(r'^\d+\.', line):
                enhanced_lines.append(line)
            
            # Regular content
            else:
                enhanced_lines.append(line)
        
        return '\n'.join(enhanced_lines)
    
    def _is_potential_header(self, line: str) -> bool:
        """Determine if a line should be treated as a header."""
        # Check for header indicators
        header_indicators = [
            'introduction', 'background', 'methodology', 'findings', 'results',
            'analysis', 'conclusion', 'recommendations', 'summary', 'overview',
            'key points', 'implications', 'next steps', 'action items'
        ]
        
        line_lower = line.lower()
        
        # Check if line is short and contains header keywords
        if len(line.split()) <= 5 and any(indicator in line_lower for indicator in header_indicators):
            return True
        
        # Check if line ends with colon
        if line.endswith(':') and len(line.split()) <= 6:
            return True
        
        return False
    
    def _compile_all_sources(self, state: MultiAgentState) -> List[str]:
        """Compile all sources from research and agent outputs."""
        sources = set()
        
        # Add research sources
        for item in state['research_data']:
            if item.get('url'):
                sources.add(item['url'])
            elif item.get('source'):
                sources.add(item['source'])
        
        # Add sources from agent outputs
        for output in state['agent_outputs'].values():
            sources.update(output.sources)
        
        return list(sources)
    
    def _create_appendix(self, state: MultiAgentState) -> str:
        """Create appendix with supporting technical details."""
        appendix_parts = []
        
        # Add research data details
        if state['research_data']:
            appendix_parts.append("### Research Data Summary")
            appendix_parts.append(f"Total research items analyzed: {len(state['research_data'])}")
            
            # Group by confidence levels
            high_conf = [item for item in state['research_data'] if item.get('confidence', 0.5) > 0.8]
            med_conf = [item for item in state['research_data'] if 0.5 < item.get('confidence', 0.5) <= 0.8]
            low_conf = [item for item in state['research_data'] if item.get('confidence', 0.5) <= 0.5]
            
            appendix_parts.append(f"- High confidence sources: {len(high_conf)}")
            appendix_parts.append(f"- Medium confidence sources: {len(med_conf)}")
            appendix_parts.append(f"- Lower confidence sources: {len(low_conf)}")
        
        # Add analysis methodology
        if state['analysis_results']:
            appendix_parts.append("\n### Analysis Methodology")
            methods = state['analysis_results'].get('methods_used', [])
            if methods:
                for method in methods:
                    appendix_parts.append(f"- {method.replace('_', ' ').title()}")
        
        # Add execution summary
        appendix_parts.append("\n### Execution Summary")
        execution_time = (state['last_updated'] - state['start_time']).total_seconds()
        appendix_parts.append(f"- Total processing time: {execution_time:.1f} seconds")
        appendix_parts.append(f"- Agents involved: {len(state['agent_outputs'])}")
        appendix_parts.append(f"- Iterations completed: {state['iteration_count']}")
        
        return "\n".join(appendix_parts) if appendix_parts else ""
    
    def _extract_content_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata about the generated content."""
        # Count words
        words = len(re.findall(r'\b\w+\b', content))
        
        # Count sections (headers)
        sections = len(re.findall(r'^#+\s', content, re.MULTILINE))
        
        # Check for key components
        has_summary = 'summary' in content.lower()
        has_recommendations = any(word in content.lower() for word in ['recommendation', 'next steps', 'action items'])
        
        # Count references
        references = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
        
        return {
            'word_count': words,
            'sections_count': sections,
            'has_summary': has_summary,
            'has_recommendations': has_recommendations,
            'references_count': references,
            'reading_time_minutes': max(1, words // 200)  # Assume 200 words per minute
        }