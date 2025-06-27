from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
# Email imports
import smtplib
import re
import os
from email.mime.text import MIMEText
# Pinecone and embedding imports
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_pinecone import PineconeVectorStore
# Web search imports
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Global variables
document_content = ""
vectorstore = None

# Initialize Pinecone with better error handling
def initialize_pinecone():
    global vectorstore
    try:
        print("🔧 Initializing Pinecone...")
        
        # Check if API key exists
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("⚠️ PINECONE_API_KEY not found in .env file")
            return False
            
        pc = pinecone.Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "company-policies")
        
        print(f"🔍 Checking for index: {index_name}")
        
        # Check if index exists
        try:
            indexes = pc.list_indexes()
            if index_name not in [index.name for index in indexes]:
                print(f"⚠️ Warning: Pinecone index '{index_name}' not found. Will use web search only.")
                return False
        except Exception as e:
            print(f"⚠️ Could not list indexes: {e}")
            return False
            
        # Initialize embeddings
        print("🔧 Initializing OpenAI embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        
        # Create vectorstore
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print(f"✅ Successfully connected to Pinecone index: {index_name}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install pinecone-client langchain-pinecone")
        return False
    except Exception as e:
        print(f"⚠️ Could not initialize Pinecone: {e}")
        print("💡 Check your .env file and Pinecone configuration")
        return False

# Initialize on startup
pinecone_available = initialize_pinecone()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def search_company_db(query: str) -> str:
    """Search the company's policy and information database for relevant content.
    
    Args:
        query: The search query to find relevant company information
    """
    global vectorstore
    
    if not vectorstore:
        return "Company database is not available. Using web search instead."
    
    try:
        print(f"🔍 Searching company database for: {query}")
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return f"No relevant information found in company database for: {query}"
        
        # Combine the relevant content
        content = []
        for i, doc in enumerate(docs, 1):
            content.append(f"Source {i}: {doc.page_content}")
        
        result = "\n\n".join(content)
        return f"Found relevant company information:\n\n{result}"
        
    except Exception as e:
        print(f"❌ Error searching company database: {e}")
        return f"Error searching company database: {str(e)}"

@tool
def search_web(query: str) -> str:
    """Search the web for information when company database doesn't have relevant content.
    
    Args:
        query: The search query for web search
    """
    try:
        # Simple web search using DuckDuckGo
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract search results
        results = []
        for result in soup.find_all('a', class_='result__a')[:3]:
            title = result.get_text().strip()
            results.append(title)
        
        if results:
            return f"Web search results for '{query}':\n" + "\n".join([f"• {r}" for r in results])
        else:
            return f"No web search results found for: {query}"
            
    except Exception as e:
        return f"Web search error: {str(e)}"

@tool
def draft_from_db(topic: str) -> str:
    """Draft a document or email using information from the company database.
    
    Args:
        topic: The topic or subject you want to draft about
    """
    global document_content
    
    # First search the company database
    db_result = search_company_db(topic)
    
    # If no relevant info found in DB, search web
    if "No relevant information found" in db_result:
        print("🔍 No company info found, searching web...")
        web_result = search_web(topic)
        source_info = f"Based on web search:\n{web_result}"
    else:
        source_info = db_result
    
    # Create a draft using the found information
    draft_prompt = f"""
    Create a professional document/email draft about: {topic}
    
    Use this information as reference:
    {source_info}
    
    Make it professional, well-structured, and appropriate for business communication.
    """
    
    # Use ChatGPT to create the draft
    model_temp = ChatOpenAI(model="gpt-4o")
    draft_response = model_temp.invoke([HumanMessage(content=draft_prompt)])
    
    document_content = draft_response.content
    
    return f"Document drafted successfully using {'company database' if 'Based on web search' not in source_info else 'web search'}!\n\nCurrent content:\n{document_content}"

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"

@tool
def send_email(_: str = "") -> str:
    """
    Sends an email if the current document content contains 'To:', 'Subject:', and body.
    The content is expected to be in format:

    To: someone@example.com
    Subject: Hello there

    This is the body of the email.
    """
    global document_content

    # Use regex to extract To, Subject, and Body
    match = re.match(r"To:\s*(.+)\s+Subject:\s*(.+?)\s*\n+(.*)", document_content, re.DOTALL)
    if not match:
        return "No valid email structure found in the document. Email not sent."

    to_address, subject, body = match.groups()

    # Load SMTP settings from environment
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        return "Missing SMTP configuration in .env file. Email not sent."

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_address

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_address], msg.as_string())

        return f"Email successfully sent to {to_address} with subject '{subject}'."

    except Exception as e:
        return f"Error sending email: {str(e)}"

# Updated tools list
tools = [update, save, send_email, search_company_db, search_web, draft_from_db]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def get_user_input():
    """Get user input with better error handling"""
    try:
        user_input = input("\n🤔 What would you like to do? (draft/update/search/save/send/quit): ").strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        return "quit"

def our_agent(state: AgentState) -> AgentState:
    global document_content

    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    - If the user want to draft a document draft a logical and professional document using the company database.
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    
    - If the user wants to save or send the email, ask them clearly whether to 'save' or 'send'.
    
    Email Guidelines:                    
    - If the user says "Write an email to [Recipients Name]", extract the [Recipient's Name].
    - NEVER ask the user for the recipient’s name — it is already in the instruction example in write an email to John, Recipient's Name is John.
    - If the draft contains [Recipient's Name], REPLACE it with the correct name.
    - Use the name in both the greeting and the "Dear:" field like:
    Dear John ...
    - Also ensure to include an email subject from the user input, like:
      Subject: [Email subject]
    - If the user asks to send an email, always include a greeting like "Dear [Recipient's Name],"                              
    - If they choose to send an email, format the document like this:

      To: recipient@example.com
      Subject: Email subject
      Dear [Recipient's Name],
                                  
    - If an email is sent ,always sign off as "Best regards, HR Team".                                                            

      Body content...
    - Remember an email should not be very long, keep it concise and to the point.
    - If they ask for a brief document, do not go beyond 100 words, a brief email, do not go beyond 30 words.
    - Do not re-ask for the subject line if it already exists.
    - Always show the current document state after modifications.

    The current document content is:\n{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")

        if user_input.strip().lower() == "send":
            to_address = input("✉️ Enter recipient email address: ")

            # Replace existing 'To:' or insert it cleanly
            if re.match(r"(?i)^to:\s*.+", document_content):
                document_content = re.sub(r"(?i)^to:\s*.+", f"To: {to_address}", document_content, count=1)
            else:
                document_content = f"To: {to_address}\n" + document_content

            print(f"\n📨 Prepared email:\n{document_content}")

            user_input = "Send the email now."

        elif user_input.strip().lower() == "save":
            filename = input("💾 Enter filename to save: ")
            user_input = f"Save the file as {filename}"

        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" 
        
    return "continue"

def print_messages(messages):
    """Print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content[:200]}{'...' if len(message.content) > 200 else ''}")

# Graph setup
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

def should_send_email(state: AgentState) -> str:
    messages = state["messages"]
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower():
            return "send"
    return "skip"

graph.add_node("send_email_node", ToolNode([send_email]))

graph.add_conditional_edges(
    "tools",
    should_send_email,
    {
        "send": "send_email_node",
        "skip": "agent"
    },
)

graph.add_edge("send_email_node", "agent")

app = graph.compile()

def run_document_agent():
    print("\n" + "="*60)
    print(" ENHANCED DRAFTER WITH COMPANY DATABASE")
    print("="*60)
    print("📚 Features: Company DB Search, Web Search, Drafting, Email Sending")
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("💡 Please check your .env file")
        return
    
    if vectorstore:
        print("✅ Company database connected")
    else:
        print("⚠️ Company database not available - will use web search fallback")
    
    print("\n🚀 Starting Drafter...")
    
    try:
        state = {"messages": []}
        
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                print_messages(step["messages"])
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Drafter stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running Drafter: {e}")
        print("💡 Check your configuration and try again")
    
    print("\n" + "="*60)
    print(" DRAFTER FINISHED")
    print("="*60)

if __name__ == "__main__":
    print("🔧 Loading environment variables...")
    load_dotenv()
    
    print("🔧 Checking OpenAI API key...")
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables!")
        print("💡 Please add it to your .env file")
        exit(1)
    
    print("✅ OpenAI API key found")
    
    try:
        run_document_agent()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()