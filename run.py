"""
Simple script to run the Agentic AI multi-agent system.
This script provides an easy way to start the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the Agentic AI Streamlit application."""
    
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    
    # Change to the Agentic_AI directory
    os.chdir(current_dir)
    
    # Set Python path to include src directory
    src_path = current_dir / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🤖 Starting Agentic AI Multi-Agent System...")
    print(f"📁 Working directory: {current_dir}")
    print("🔧 Make sure you have:")
    print("   1. Installed all requirements: pip install -r requirements.txt")
    print("   2. Set up your .env file with API keys")
    print("   3. Configured your LLM provider credentials")
    print("\n🚀 Launching Streamlit application...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8502",
            "--server.address=localhost",
            "--theme.base=light"
        ], check=True)
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down Agentic AI system...")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()