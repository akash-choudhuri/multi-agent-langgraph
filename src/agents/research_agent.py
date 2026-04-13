"""
Research Agent for multi-agent system.
Specializes in information gathering, web search, and knowledge synthesis.
"""

import requests
import json
import re
from typing import Dict, Any, List
from urllib.parse import urlparse
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage
from bs4 import BeautifulSoup

from .base_agent import BaseAgent, AgentConfig
from ..core.state import MultiAgentState, AgentType, StateManager


class ResearchAgent(BaseAgent):
    """Agent specialized in research and information gathering."""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                agent_type=AgentType.RESEARCHER,
                model_name="gpt-4",
                temperature=0.1,  # Lower temperature for factual research
                max_tokens=3000
            )
        super().__init__(config)
    
    def get_system_message(self) -> str:
        """Get the system message for the research agent."""
        return """You are a Research Agent, an expert at gathering, analyzing, and synthesizing information from various sources. Your role is to:

1. INFORMATION GATHERING: Collect relevant data and insights related to the task
2. SOURCE EVALUATION: Assess credibility and relevance of information sources
3. KNOWLEDGE SYNTHESIS: Combine information from multiple sources into coherent insights
4. FACT VERIFICATION: Ensure accuracy and reliability of gathered information
5. STRUCTURED OUTPUT: Present findings in an organized, actionable format

Key Capabilities:
- Web search and content analysis
- Document processing and extraction
- Data validation and cross-referencing
- Source citation and attribution
- Knowledge gap identification

Guidelines:
- Prioritize authoritative and recent sources
- Provide clear citations and references
- Identify conflicting information and uncertainties
- Suggest additional research directions if needed
- Format output for easy consumption by other agents

Output Format:
- Executive Summary
- Key Findings (with confidence levels)
- Sources and References
- Limitations and Gaps
- Recommendations for further research"""
    
    def process_task(self, state: MultiAgentState) -> MultiAgentState:
        """Process research task and update state."""
        # Create research-focused prompt
        research_context = self._analyze_research_requirements(state)
        prompt = self._create_prompt(state, research_context)
        
        # Execute research
        research_results = self._conduct_research(state['task_objective'], state['original_query'])
        
        # Generate research report
        messages = [
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=f"{prompt}\n\nBased on the research findings below, provide a comprehensive research report:\n\n{research_results}")
        ]
        
        research_report = self._execute_llm_call(messages)
        
        # Extract structured data from the report
        structured_data = self._extract_structured_data(research_report, research_results)
        
        # Create agent output
        output = self.create_agent_output(
            content=research_report,
            sources=structured_data.get('sources', []),
            metadata={
                'research_method': 'web_search_and_analysis',
                'sources_count': len(structured_data.get('sources', [])),
                'key_findings': structured_data.get('key_findings', []),
                'confidence_level': structured_data.get('confidence', 0.8)
            }
        )
        
        # Update state
        state = StateManager.update_agent_output(state, AgentType.RESEARCHER, output)
        
        # Add research data to shared knowledge
        for item in structured_data.get('research_items', []):
            state = StateManager.add_research_data(state, item)
        
        return state
    
    def _analyze_research_requirements(self, state: MultiAgentState) -> str:
        """Analyze what research is needed based on the task."""
        context_parts = []
        
        # Analyze task objective for research keywords
        objective = state['task_objective']
        query = state['original_query']
        
        context_parts.append("RESEARCH ANALYSIS:")
        context_parts.append(f"Primary Focus: {objective}")
        context_parts.append(f"Original Query: {query}")
        
        # Identify research domains
        domains = self._identify_research_domains(objective, query)
        if domains:
            context_parts.append(f"Research Domains: {', '.join(domains)}")
        
        # Suggest research approach
        approach = self._suggest_research_approach(objective, query)
        context_parts.append(f"Recommended Approach: {approach}")
        
        return "\n".join(context_parts)
    
    def _identify_research_domains(self, objective: str, query: str) -> List[str]:
        """Identify relevant research domains."""
        text = f"{objective} {query}".lower()
        
        domain_keywords = {
            'technology': ['ai', 'machine learning', 'software', 'tech', 'digital', 'algorithm'],
            'business': ['market', 'business', 'strategy', 'economic', 'financial', 'commercial'],
            'science': ['research', 'study', 'analysis', 'data', 'experiment', 'scientific'],
            'health': ['health', 'medical', 'healthcare', 'clinical', 'treatment', 'disease'],
            'education': ['education', 'learning', 'training', 'academic', 'curriculum', 'teaching'],
            'social': ['social', 'community', 'society', 'culture', 'people', 'human']
        }
        
        identified_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                identified_domains.append(domain)
        
        return identified_domains
    
    def _suggest_research_approach(self, objective: str, query: str) -> str:
        """Suggest the most appropriate research approach."""
        text = f"{objective} {query}".lower()
        
        if any(word in text for word in ['recent', 'latest', 'current', 'new']):
            return "Focus on recent developments and current trends"
        elif any(word in text for word in ['compare', 'versus', 'difference']):
            return "Comparative analysis with multiple sources"
        elif any(word in text for word in ['trend', 'pattern', 'evolution']):
            return "Historical analysis and trend identification"
        elif any(word in text for word in ['how', 'method', 'process', 'step']):
            return "Process-oriented research with practical examples"
        else:
            return "Comprehensive overview with multiple perspectives"
    
    def _conduct_research(self, objective: str, query: str) -> str:
        """Conduct research using available methods."""
        research_results = []
        
        # Simulated web search results (in a real implementation, you'd use actual search APIs)
        search_results = self._simulate_web_search(query)
        
        research_results.append("=== WEB SEARCH RESULTS ===")
        research_results.extend(search_results)
        
        # Add domain-specific knowledge
        domain_knowledge = self._get_domain_knowledge(objective, query)
        if domain_knowledge:
            research_results.append("\n=== DOMAIN KNOWLEDGE ===")
            research_results.append(domain_knowledge)
        
        # Add structured research questions
        research_questions = self._generate_research_questions(objective, query)
        research_results.append("\n=== KEY RESEARCH QUESTIONS ===")
        research_results.extend(research_questions)
        
        return "\n".join(research_results)
    
    def _simulate_web_search(self, query: str) -> List[str]:
        """Simulate web search results (replace with actual search API in production)."""
        # This is a simplified simulation - in production, use actual search APIs
        mock_results = [
            {
                'title': f'Research Study: {query}',
                'url': 'https://example-research.com/study1',
                'snippet': f'Comprehensive analysis of {query} covering recent developments and key insights. Published by leading researchers in the field.',
                'date': '2024-01-15'
            },
            {
                'title': f'{query}: Industry Report 2024',
                'url': 'https://industry-reports.com/report2024',
                'snippet': f'Latest industry trends and market analysis related to {query}. Includes statistical data and expert opinions.',
                'date': '2024-01-10'
            },
            {
                'title': f'Expert Guide to {query}',
                'url': 'https://expert-guides.com/guide',
                'snippet': f'Detailed guide covering all aspects of {query} with practical examples and case studies.',
                'date': '2023-12-20'
            }
        ]
        
        formatted_results = []
        for result in mock_results:
            formatted_results.append(f"Title: {result['title']}")
            formatted_results.append(f"URL: {result['url']}")
            formatted_results.append(f"Snippet: {result['snippet']}")
            formatted_results.append(f"Date: {result['date']}")
            formatted_results.append("---")
        
        return formatted_results
    
    def _get_domain_knowledge(self, objective: str, query: str) -> str:
        """Provide domain-specific knowledge based on the query."""
        text = f"{objective} {query}".lower()
        
        knowledge_base = {
            'ai': "AI and machine learning technologies continue to evolve rapidly. Key areas include natural language processing, computer vision, and generative AI. Current trends focus on large language models, ethical AI, and practical applications.",
            'business': "Modern business strategies emphasize digital transformation, customer experience, and data-driven decision making. Market dynamics are influenced by technological disruption and changing consumer behaviors.",
            'technology': "Technology trends include cloud computing, edge computing, cybersecurity, and sustainable tech solutions. Digital transformation remains a key priority for organizations worldwide.",
            'health': "Healthcare is experiencing transformation through digital health technologies, personalized medicine, and AI-driven diagnostics. Telemedicine and remote monitoring have gained significant adoption."
        }
        
        for domain, knowledge in knowledge_base.items():
            if domain in text:
                return f"{domain.upper()} DOMAIN KNOWLEDGE: {knowledge}"
        
        return ""
    
    def _generate_research_questions(self, objective: str, query: str) -> List[str]:
        """Generate relevant research questions."""
        questions = [
            f"What are the current trends in {query}?",
            f"What are the key challenges and opportunities related to {objective}?",
            f"What do experts say about {query}?",
            f"What are the practical applications and examples?",
            f"What are the future implications and predictions?"
        ]
        
        return [f"• {q}" for q in questions]
    
    def _extract_structured_data(self, research_report: str, raw_results: str) -> Dict[str, Any]:
        """Extract structured data from research results."""
        # Extract sources using regex patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        sources = re.findall(url_pattern, raw_results)
        
        # Extract key findings (look for bullet points or numbered lists)
        findings_pattern = r'(?:^|\n)(?:[•\-\*]|\d+\.)\s*(.+)'
        findings = re.findall(findings_pattern, research_report, re.MULTILINE)
        
        # Create research items for shared knowledge
        research_items = []
        for i, source in enumerate(sources[:3]):  # Limit to first 3 sources
            research_items.append({
                'title': f'Research Source {i+1}',
                'url': source,
                'summary': findings[i] if i < len(findings) else 'Research finding',
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.8
            })
        
        return {
            'sources': sources,
            'key_findings': findings[:5],  # Top 5 findings
            'research_items': research_items,
            'confidence': 0.8
        }