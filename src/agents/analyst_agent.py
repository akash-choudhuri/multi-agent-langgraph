"""
Analyst Agent for multi-agent system.
Specializes in data analysis, pattern recognition, and insight generation.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
import re

from langchain.schema import HumanMessage, SystemMessage

from .base_agent import BaseAgent, AgentConfig
from ..core.state import MultiAgentState, AgentType, StateManager


class AnalystAgent(BaseAgent):
    """Agent specialized in data analysis and insight generation."""
    
    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                agent_type=AgentType.ANALYST,
                model_name="gpt-4",
                temperature=0.2,  # Low temperature for analytical consistency
                max_tokens=3000
            )
        super().__init__(config)
    
    def get_system_message(self) -> str:
        """Get the system message for the analyst agent."""
        return """You are an Analyst Agent, an expert in data analysis, pattern recognition, and insight generation. Your role is to:

1. DATA ANALYSIS: Process and analyze information from research and other sources
2. PATTERN IDENTIFICATION: Recognize trends, correlations, and anomalies in data
3. INSIGHT GENERATION: Draw meaningful conclusions from available information
4. STATISTICAL REASONING: Apply analytical methods to validate findings
5. PREDICTIVE ANALYSIS: Identify future trends and implications

Key Capabilities:
- Quantitative and qualitative analysis
- Trend analysis and forecasting
- Comparative analysis and benchmarking
- Risk assessment and opportunity identification
- Data visualization recommendations
- Statistical validation

Guidelines:
- Base conclusions on evidence from available data
- Quantify insights where possible with confidence levels
- Identify limitations and assumptions in analysis
- Provide actionable recommendations
- Highlight key metrics and indicators
- Consider multiple perspectives and scenarios

Output Format:
- Executive Summary of Analysis
- Key Insights and Findings
- Supporting Data and Evidence
- Trends and Patterns Identified
- Recommendations and Next Steps
- Confidence Levels and Limitations"""
    
    def process_task(self, state: MultiAgentState) -> MultiAgentState:
        """Process analytical task and update state."""
        # Analyze available data
        analysis_context = self._prepare_analysis_context(state)
        prompt = self._create_prompt(state, analysis_context)
        
        # Perform analysis
        analytical_insights = self._conduct_analysis(state)
        
        # Generate analysis report
        messages = [
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=f"{prompt}\n\nBased on the available data and research, provide a comprehensive analysis:\n\n{analytical_insights}")
        ]
        
        analysis_report = self._execute_llm_call(messages)
        
        # Extract structured analysis results
        structured_analysis = self._extract_analysis_results(analysis_report, state)
        
        # Create agent output
        output = self.create_agent_output(
            content=analysis_report,
            sources=self._get_data_sources(state),
            metadata={
                'analysis_type': 'comprehensive_insight_generation',
                'data_points_analyzed': structured_analysis.get('data_points_count', 0),
                'insights_generated': len(structured_analysis.get('insights', [])),
                'confidence_level': structured_analysis.get('overall_confidence', 0.75),
                'analysis_methods': structured_analysis.get('methods_used', [])
            }
        )
        
        # Update state with analysis results
        state = StateManager.update_agent_output(state, AgentType.ANALYST, output)
        state = StateManager.update_analysis_results(state, structured_analysis)
        
        return state
    
    def _prepare_analysis_context(self, state: MultiAgentState) -> str:
        """Prepare context for analysis based on available data."""
        context_parts = []
        
        context_parts.append("ANALYTICAL CONTEXT:")
        context_parts.append(f"Analysis Objective: {state['task_objective']}")
        
        # Analyze research data availability
        research_data_summary = self._summarize_research_data(state['research_data'])
        context_parts.append(f"Research Data Summary: {research_data_summary}")
        
        # Identify analysis opportunities
        analysis_opportunities = self._identify_analysis_opportunities(state)
        context_parts.append(f"Analysis Opportunities: {', '.join(analysis_opportunities)}")
        
        # Suggest analytical approaches
        approaches = self._suggest_analytical_approaches(state)
        context_parts.append(f"Recommended Approaches: {', '.join(approaches)}")
        
        return "\n".join(context_parts)
    
    def _summarize_research_data(self, research_data: List[Dict[str, Any]]) -> str:
        """Summarize available research data for analysis."""
        if not research_data:
            return "No research data available for analysis"
        
        summary_parts = []
        summary_parts.append(f"Total items: {len(research_data)}")
        
        # Categorize data types
        data_types = {}
        for item in research_data:
            item_type = item.get('type', 'general')
            data_types[item_type] = data_types.get(item_type, 0) + 1
        
        if data_types:
            type_summary = ", ".join([f"{k}: {v}" for k, v in data_types.items()])
            summary_parts.append(f"Data types: {type_summary}")
        
        # Check for quantitative data
        has_numbers = any(self._contains_numerical_data(item) for item in research_data)
        if has_numbers:
            summary_parts.append("Contains quantitative data suitable for statistical analysis")
        
        return "; ".join(summary_parts)
    
    def _contains_numerical_data(self, data_item: Dict[str, Any]) -> bool:
        """Check if a data item contains numerical information."""
        text = str(data_item.get('summary', '')) + str(data_item.get('title', ''))
        # Look for numbers, percentages, and statistical indicators
        number_patterns = [r'\d+\.?\d*%', r'\d+\.?\d*\s*(million|billion|thousand)', r'\d+\.?\d*']
        return any(re.search(pattern, text) for pattern in number_patterns)
    
    def _identify_analysis_opportunities(self, state: MultiAgentState) -> List[str]:
        """Identify potential analysis opportunities."""
        opportunities = []
        
        # Check for comparative analysis
        if len(state['research_data']) >= 2:
            opportunities.append("comparative_analysis")
        
        # Check for trend analysis
        query = state['original_query'].lower()
        if any(word in query for word in ['trend', 'change', 'growth', 'decline', 'evolution']):
            opportunities.append("trend_analysis")
        
        # Check for impact analysis
        if any(word in query for word in ['impact', 'effect', 'influence', 'consequence']):
            opportunities.append("impact_analysis")
        
        # Check for risk analysis
        if any(word in query for word in ['risk', 'challenge', 'problem', 'threat']):
            opportunities.append("risk_analysis")
        
        # Check for opportunity analysis
        if any(word in query for word in ['opportunity', 'potential', 'benefit', 'advantage']):
            opportunities.append("opportunity_analysis")
        
        return opportunities if opportunities else ["general_analysis"]
    
    def _suggest_analytical_approaches(self, state: MultiAgentState) -> List[str]:
        """Suggest appropriate analytical approaches."""
        approaches = []
        
        # Based on data availability
        if state['research_data']:
            approaches.append("evidence_based_analysis")
        
        # Based on task requirements
        for req in state['task_requirements']:
            req_text = req.description.lower()
            if 'compare' in req_text:
                approaches.append("comparative_method")
            if 'analyze' in req_text:
                approaches.append("systematic_analysis")
            if 'predict' in req_text:
                approaches.append("predictive_modeling")
        
        return approaches if approaches else ["comprehensive_review"]
    
    def _conduct_analysis(self, state: MultiAgentState) -> str:
        """Conduct comprehensive analysis of available data."""
        analysis_results = []
        
        # Quantitative analysis if applicable
        quantitative_results = self._perform_quantitative_analysis(state['research_data'])
        if quantitative_results:
            analysis_results.append("=== QUANTITATIVE ANALYSIS ===")
            analysis_results.append(quantitative_results)
        
        # Qualitative analysis
        qualitative_results = self._perform_qualitative_analysis(state)
        analysis_results.append("\n=== QUALITATIVE ANALYSIS ===")
        analysis_results.append(qualitative_results)
        
        # Pattern analysis
        pattern_analysis = self._identify_patterns(state)
        analysis_results.append("\n=== PATTERN ANALYSIS ===")
        analysis_results.append(pattern_analysis)
        
        # Comparative analysis
        if len(state['research_data']) >= 2:
            comparative_analysis = self._perform_comparative_analysis(state['research_data'])
            analysis_results.append("\n=== COMPARATIVE ANALYSIS ===")
            analysis_results.append(comparative_analysis)
        
        # Risk and opportunity assessment
        risk_opportunity = self._assess_risks_and_opportunities(state)
        analysis_results.append("\n=== RISK & OPPORTUNITY ASSESSMENT ===")
        analysis_results.append(risk_opportunity)
        
        return "\n".join(analysis_results)
    
    def _perform_quantitative_analysis(self, research_data: List[Dict[str, Any]]) -> str:
        """Perform quantitative analysis on available data."""
        if not research_data:
            return ""
        
        # Extract numerical data
        numerical_data = []
        for item in research_data:
            numbers = self._extract_numbers(str(item))
            numerical_data.extend(numbers)
        
        if not numerical_data:
            return "No significant quantitative data available for statistical analysis."
        
        # Perform basic statistical analysis
        try:
            data_array = np.array(numerical_data)
            stats = {
                'count': len(data_array),
                'mean': np.mean(data_array),
                'median': np.median(data_array),
                'std': np.std(data_array),
                'min': np.min(data_array),
                'max': np.max(data_array)
            }
            
            results = []
            results.append(f"Statistical Summary of {stats['count']} data points:")
            results.append(f"- Mean: {stats['mean']:.2f}")
            results.append(f"- Median: {stats['median']:.2f}")
            results.append(f"- Standard Deviation: {stats['std']:.2f}")
            results.append(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            # Add interpretation
            if stats['std'] / stats['mean'] > 0.5:  # High variability
                results.append("- High variability detected in the data")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Quantitative analysis error: {str(e)}"
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Extract various number formats
        patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\d+\.?\d*',   # Regular numbers
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Remove percentage sign and convert
                    num_str = match.replace('%', '')
                    numbers.append(float(num_str))
                except ValueError:
                    continue
        
        return numbers
    
    def _perform_qualitative_analysis(self, state: MultiAgentState) -> str:
        """Perform qualitative analysis of available information."""
        results = []
        
        # Analyze themes and topics
        themes = self._identify_themes(state)
        results.append(f"Key Themes Identified: {', '.join(themes)}")
        
        # Analyze sentiment and tone
        sentiment = self._analyze_sentiment(state)
        results.append(f"Overall Sentiment: {sentiment}")
        
        # Analyze stakeholder perspectives
        perspectives = self._identify_stakeholder_perspectives(state)
        if perspectives:
            results.append(f"Stakeholder Perspectives: {', '.join(perspectives)}")
        
        return "\n".join(results)
    
    def _identify_themes(self, state: MultiAgentState) -> List[str]:
        """Identify key themes from research data."""
        all_text = state['original_query'] + " " + state['task_objective']
        
        # Add research data text
        for item in state['research_data']:
            all_text += " " + str(item.get('summary', '')) + " " + str(item.get('title', ''))
        
        # Simple theme extraction based on keyword frequency
        words = re.findall(r'\b\w+\b', all_text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top themes
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        themes = [word for word, freq in sorted_words[:5] if freq > 1]
        
        return themes
    
    def _analyze_sentiment(self, state: MultiAgentState) -> str:
        """Analyze overall sentiment of the information."""
        # Simple sentiment analysis based on keyword presence
        positive_words = ['opportunity', 'benefit', 'advantage', 'growth', 'success', 'improvement']
        negative_words = ['risk', 'challenge', 'problem', 'decline', 'threat', 'difficulty']
        
        all_text = state['original_query'] + " " + state['task_objective']
        for item in state['research_data']:
            all_text += " " + str(item.get('summary', ''))
        
        all_text = all_text.lower()
        
        positive_count = sum(1 for word in positive_words if word in all_text)
        negative_count = sum(1 for word in negative_words if word in all_text)
        
        if positive_count > negative_count:
            return "Predominantly Positive"
        elif negative_count > positive_count:
            return "Predominantly Negative"
        else:
            return "Neutral/Balanced"
    
    def _identify_stakeholder_perspectives(self, state: MultiAgentState) -> List[str]:
        """Identify different stakeholder perspectives."""
        perspectives = []
        
        query_lower = state['original_query'].lower()
        objective_lower = state['task_objective'].lower()
        
        # Check for business perspective
        if any(word in query_lower + objective_lower for word in ['business', 'company', 'market', 'customer']):
            perspectives.append("Business")
        
        # Check for technical perspective
        if any(word in query_lower + objective_lower for word in ['technical', 'technology', 'engineering', 'development']):
            perspectives.append("Technical")
        
        # Check for user/consumer perspective
        if any(word in query_lower + objective_lower for word in ['user', 'consumer', 'customer', 'people']):
            perspectives.append("User/Consumer")
        
        # Check for regulatory perspective
        if any(word in query_lower + objective_lower for word in ['regulation', 'compliance', 'legal', 'policy']):
            perspectives.append("Regulatory")
        
        return perspectives
    
    def _identify_patterns(self, state: MultiAgentState) -> str:
        """Identify patterns in the available data."""
        patterns = []
        
        # Look for temporal patterns
        if any(word in state['original_query'].lower() for word in ['trend', 'over time', 'recent', 'evolution']):
            patterns.append("Temporal progression pattern identified")
        
        # Look for cause-effect patterns
        if any(word in state['original_query'].lower() for word in ['because', 'due to', 'impact', 'effect']):
            patterns.append("Cause-effect relationships present")
        
        # Look for comparison patterns
        if any(word in state['original_query'].lower() for word in ['versus', 'compared to', 'difference', 'better']):
            patterns.append("Comparative patterns detected")
        
        return "; ".join(patterns) if patterns else "No specific patterns identified"
    
    def _perform_comparative_analysis(self, research_data: List[Dict[str, Any]]) -> str:
        """Perform comparative analysis of research items."""
        if len(research_data) < 2:
            return "Insufficient data for comparative analysis"
        
        comparisons = []
        comparisons.append(f"Comparing {len(research_data)} research items:")
        
        # Compare by confidence levels
        confidences = [item.get('confidence', 0.5) for item in research_data]
        avg_confidence = sum(confidences) / len(confidences)
        comparisons.append(f"- Average confidence level: {avg_confidence:.2f}")
        
        # Compare by data freshness
        timestamps = [item.get('timestamp') for item in research_data if item.get('timestamp')]
        if timestamps:
            comparisons.append(f"- Data spans {len(timestamps)} time points")
        
        return "\n".join(comparisons)
    
    def _assess_risks_and_opportunities(self, state: MultiAgentState) -> str:
        """Assess risks and opportunities based on analysis."""
        assessment = []
        
        # Risk assessment
        risks = self._identify_risks(state)
        if risks:
            assessment.append("RISKS IDENTIFIED:")
            assessment.extend([f"- {risk}" for risk in risks])
        
        # Opportunity assessment
        opportunities = self._identify_opportunities(state)
        if opportunities:
            assessment.append("\nOPPORTUNITIES IDENTIFIED:")
            assessment.extend([f"- {opp}" for opp in opportunities])
        
        return "\n".join(assessment) if assessment else "No specific risks or opportunities identified"
    
    def _identify_risks(self, state: MultiAgentState) -> List[str]:
        """Identify potential risks from the analysis."""
        risks = []
        
        # Check for uncertainty indicators
        if len(state['research_data']) < 3:
            risks.append("Limited data availability may affect conclusion reliability")
        
        # Check for conflicting information
        if len(state['errors']) > 0:
            risks.append("Errors encountered during data processing")
        
        return risks
    
    def _identify_opportunities(self, state: MultiAgentState) -> List[str]:
        """Identify potential opportunities from the analysis."""
        opportunities = []
        
        # Check for data richness
        if len(state['research_data']) > 5:
            opportunities.append("Rich data set enables comprehensive analysis")
        
        # Check for multi-perspective analysis
        if len(state['agent_outputs']) > 1:
            opportunities.append("Multi-agent collaboration provides diverse perspectives")
        
        return opportunities
    
    def _get_data_sources(self, state: MultiAgentState) -> List[str]:
        """Get list of data sources used in analysis."""
        sources = []
        
        # Add research sources
        for item in state['research_data']:
            source = item.get('url') or item.get('source', 'Research Data')
            if source not in sources:
                sources.append(source)
        
        return sources
    
    def _extract_analysis_results(self, analysis_report: str, state: MultiAgentState) -> Dict[str, Any]:
        """Extract structured results from analysis report."""
        # Extract insights (look for bullet points or numbered items)
        insights_pattern = r'(?:^|\n)(?:[•\-\*]|\d+\.)\s*(.+)'
        insights = re.findall(insights_pattern, analysis_report, re.MULTILINE)
        
        # Calculate overall confidence based on data quality
        data_quality_score = min(1.0, len(state['research_data']) / 5.0)  # More data = higher confidence
        error_penalty = len(state['errors']) * 0.1
        overall_confidence = max(0.1, data_quality_score - error_penalty)
        
        return {
            'insights': insights[:10],  # Top 10 insights
            'data_points_count': len(state['research_data']),
            'overall_confidence': overall_confidence,
            'methods_used': ['qualitative_analysis', 'pattern_recognition', 'comparative_analysis'],
            'analysis_timestamp': datetime.now().isoformat(),
            'key_metrics': {
                'research_items_analyzed': len(state['research_data']),
                'insights_generated': len(insights),
                'confidence_score': overall_confidence
            }
        }