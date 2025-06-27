#!/usr/bin/env python3
"""
Diagnostic script to check if all dependencies and configurations are correct
Run this before running the main Drafter script
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Python Version Check:")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️ Warning: Python 3.8+ recommended")
    else:
        print("   ✅ Python version OK")
    print()

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Dependencies Check:")
    
    required_packages = [
        "dotenv",
        "langchain_core", 
        "langchain_openai",
        "langgraph",
        "openai"
    ]
    
    optional_packages = [
        "pinecone",
        "langchain_pinecone", 
        "requests",
        "beautifulsoup4"
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING (Required)")
            missing_required.append(package)
    
    # Check optional packages
    for package in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ⚠️ {package} - MISSING (Optional - reduces functionality)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\n❌ Install required packages: pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n💡 Install optional packages for full functionality: pip install {' '.join(missing_optional)}")
    
    print()
    return len(missing_required) == 0

def check_env_file():
    """Check .env file and environment variables"""
    print("🔧 Environment Configuration Check:")
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file found")
    else:
        print("   ⚠️ .env file not found - create one with your API keys")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("   ✅ dotenv loaded")
    except ImportError:
        print("   ❌ python-dotenv not installed")
        return False
    
    # Check critical environment variables
    env_vars = {
        "OPENAI_API_KEY": "Required for AI functionality",
        "PINECONE_API_KEY": "Optional - for company database",
        "PINECONE_INDEX_NAME": "Optional - defaults to 'company-policies'",
        "SMTP_SERVER": "Optional - for email sending",
        "SMTP_USER": "Optional - for email sending",
        "SMTP_PASSWORD": "Optional - for email sending"
    }
    
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"   ✅ {var}: {masked_value}")
        else:
            status = "❌" if "Required" in description else "⚠️"
            print(f"   {status} {var}: Not set - {description}")
    
    print()
    return bool(os.getenv("OPENAI_API_KEY"))

def test_basic_imports():
    """Test basic imports that the main script needs"""
    print("🧪 Basic Import Test:")
    
    try:
        from langchain_core.messages import BaseMessage, HumanMessage
        print("   ✅ LangChain core imports")
    except ImportError as e:
        print(f"   ❌ LangChain core import failed: {e}")
        return False
    
    try:
        from langchain_openai import ChatOpenAI
        print("   ✅ OpenAI integration")
    except ImportError as e:
        print(f"   ❌ OpenAI integration failed: {e}")
        return False
    
    try:
        from langgraph.graph import StateGraph
        print("   ✅ LangGraph")
    except ImportError as e:
        print(f"   ❌ LangGraph import failed: {e}")
        return False
    
    print()
    return True

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔗 OpenAI Connection Test:")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv("OPENAI_API_KEY"):
            print("   ❌ OPENAI_API_KEY not set")
            return False
        
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model="gpt-4o", max_tokens=10)
        
        from langchain_core.messages import HumanMessage
        response = model.invoke([HumanMessage(content="Say 'Hello'")])
        
        print("   ✅ OpenAI connection successful")
        print(f"   📝 Test response: {response.content}")
        return True
        
    except Exception as e:
        print(f"   ❌ OpenAI connection failed: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("🔍 DRAFTER DIAGNOSTIC TOOL")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_env_file,
        test_basic_imports,
        test_openai_connection
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"❌ Check failed with error: {e}")
            results.append(False)
    
    print("=" * 50)
    print("📊 SUMMARY:")
    
    if all(r is not False for r in results):
        print("✅ All checks passed! Your system should be ready to run Drafter.")
    else:
        print("⚠️ Some issues found. Please address the problems above before running Drafter.")
        print("\n💡 Common solutions:")
        print("   • pip install -r requirements.txt")
        print("   • Create .env file with your API keys")
        print("   • Check your OpenAI API key is valid")
    
    print("\n🚀 To run Drafter: python Drafter_and_sender.py")

if __name__ == "__main__":
    main()