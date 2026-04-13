# 🤖 Agentic AI Setup Guide

This comprehensive guide will walk you through setting up and running the Agentic AI multi-agent system using LangGraph.

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)
- At least 8GB of free RAM (recommended 16GB+)
- OpenAI API key or Anthropic API key (for LLM access)

## 🛠️ Installation Steps

### 1. Navigate to the Project Directory
```bash
cd /path/to/your/Agentic_AI/project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys (REQUIRED)
# nano .env  # or use any text editor
```

**Important:** You must configure at least one LLM provider in your `.env` file:

```env
# Required: At least one LLM provider
OPENAI_API_KEY=your_openai_api_key_here
# OR
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: For enhanced web search capabilities
SERPER_API_KEY=your_serper_api_key
SERPAPI_API_KEY=your_serpapi_key
```

### 5. Run the Application
```bash
# Option 1: Using the run script (recommended)
python run.py

# Option 2: Direct Streamlit command
streamlit run app.py --server.port 8502
```

## 🌐 Accessing the Application

Once the application starts, you can access it at:
- **URL**: http://localhost:8502
- The application will automatically open in your default web browser

## 🤖 Using the Multi-Agent System

### Step 1: Task Definition
1. Go to the **"Task Execution"** tab
2. Enter your query or question in the text area
3. Optionally specify an objective if different from the query
4. Add specific requirements (one per line)

### Step 2: Configure Parameters
1. Set task priority (low, medium, high, urgent)
2. Adjust maximum iterations (1-5)
3. Click **"Execute Task"** to start the multi-agent workflow

### Step 3: Monitor Progress
1. Watch real-time status updates as agents work
2. View agent collaboration in the progress indicators
3. Wait for completion notification

### Step 4: Review Results
1. Go to the **"Results Dashboard"** tab to see:
   - Final generated content
   - Quality assessment scores
   - Individual agent outputs
   - Execution summary and metrics
2. Download results as markdown files
3. Review detailed agent contributions

### Step 5: Track History
1. Check the **"Task History"** tab for:
   - Previous task executions
   - Performance metrics over time
   - Historical quality scores

## ⚙️ System Architecture

### Multi-Agent Workflow
The system uses a LangGraph-based workflow with the following agents:

1. **🔍 Research Agent**: Gathers information and conducts analysis
2. **📊 Analyst Agent**: Processes data and generates insights
3. **✍️ Writer Agent**: Creates comprehensive written content
4. **🔍 Reviewer Agent**: Assesses quality and provides feedback
5. **✨ Finalizer Agent**: Prepares final output and summary

### Workflow Process
```
Query Input → Research → Analysis → Writing → Review → Finalization → Output
     ↑                                                    ↓
     └─────────────── Iteration Loop (if needed) ────────┘
```

## 🔧 Configuration Options

### Model Configuration
Edit your `.env` file to customize models:

```env
# Default models for each agent
RESEARCH_AGENT_MODEL=gpt-4
ANALYST_AGENT_MODEL=gpt-4
WRITER_AGENT_MODEL=gpt-4
REVIEWER_AGENT_MODEL=gpt-3.5-turbo

# Global settings
MAX_CONCURRENT_AGENTS=5
TASK_TIMEOUT=300
MAX_ITERATIONS=3
```

### Agent Behavior
You can customize agent behavior by modifying the configuration in:
- `src/workflows/multi_agent_workflow.py`
- Individual agent files in `src/agents/`

### Performance Tuning
```env
# Memory and performance settings
LOG_LEVEL=INFO
DEBUG_MODE=False

# Streamlit configuration
STREAMLIT_PORT=8502
API_PORT=8000
```

## 🔍 Sample Use Cases

### 1. Research Report Generation
**Query:** "Analyze the impact of artificial intelligence on healthcare"
**Requirements:**
- Gather recent studies and data
- Analyze trends and implications
- Generate comprehensive report with recommendations

### 2. Market Analysis
**Query:** "Compare cloud computing platforms for enterprise use"
**Requirements:**
- Research major cloud providers
- Analyze features and pricing
- Provide decision framework

### 3. Technical Documentation
**Query:** "Create a guide for implementing microservices architecture"
**Requirements:**
- Research best practices
- Include practical examples
- Address common challenges

### 4. Strategic Planning
**Query:** "Develop a digital transformation strategy for traditional retail"
**Requirements:**
- Analyze market trends
- Identify opportunities and risks
- Create actionable roadmap

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "Module not found" errors
```bash
# Ensure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. API Key errors
- Verify your API keys are correctly set in the `.env` file
- Ensure you have sufficient API credits
- Check API key permissions and rate limits

#### 3. Memory issues
- Use a machine with at least 8GB RAM
- Close other memory-intensive applications
- Reduce `MAX_CONCURRENT_AGENTS` in `.env`

#### 4. Slow performance
- Check your internet connection (agents need API access)
- Reduce `MAX_ITERATIONS` for faster execution
- Use smaller/faster models in configuration

#### 5. Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8503
```

#### 6. Agent execution failures
- Check logs in the terminal for detailed error messages
- Verify all dependencies are installed correctly
- Ensure API quotas haven't been exceeded

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Network**: Stable internet connection
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 10GB free space
- **GPU**: Not required but beneficial for local models

## 🔄 Updating the System

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Application Data
If you encounter persistent issues:
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/
# Clear any local data files if needed
```

## 📈 Performance Optimization

### For Better Quality
1. Use GPT-4 for all agents (higher API costs)
2. Increase `MAX_ITERATIONS` for more refinement
3. Add detailed requirements in your tasks
4. Use higher priority settings for important tasks

### For Faster Execution
1. Use GPT-3.5-turbo for non-critical agents
2. Set `MAX_ITERATIONS=1` for simple tasks
3. Reduce `TASK_TIMEOUT` values
4. Use lower priority for exploratory tasks

### For Cost Optimization
1. Use mix of GPT-4 (for critical agents) and GPT-3.5-turbo
2. Optimize your queries to be specific and focused
3. Monitor API usage through provider dashboards
4. Set appropriate timeout values

## 🧪 Testing the System

### Quick Test
1. Start the application
2. Enter query: "What are the benefits of renewable energy?"
3. Use default settings and execute
4. Verify all agents run successfully
5. Check output quality and completeness

### Advanced Testing
```bash
# Run with different priorities and iterations
# Test various query types:
# - Analytical questions
# - Research requests
# - Comparative analyses
# - Technical documentation needs
```

## 📞 Getting Help

### Debug Information
When reporting issues, include:
1. Python version (`python --version`)
2. Operating system
3. Error messages from terminal
4. Contents of `.env` file (without API keys)
5. Steps to reproduce the issue

### Performance Monitoring
- Monitor agent execution times
- Check quality assessment scores
- Review error/warning messages
- Track memory usage during execution

---

**🎉 You're all set! Start exploring the power of multi-agent AI collaboration! 🤖✨**