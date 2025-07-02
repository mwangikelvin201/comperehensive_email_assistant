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
# Email reading imports
import imaplib
import email
from email.header import decode_header
import time
from datetime import datetime, timedelta
import json
import threading
from email.utils import parseaddr

load_dotenv()

# Global variables
document_content = ""
vectorstore = None
email_monitor_running = False
processed_emails = set()  # Track processed email UIDs
auto_response_rules = {}  # Store auto-response rules

# Email importance keywords with weights
IMPORTANCE_KEYWORDS = {
    # High importance (weight 10)
    'urgent': 10, 'emergency': 10, 'critical': 10, 'asap': 10,
    # Company-related (weight 8-9)
    'policy': 9, 'policies': 9, 'hr': 8, 'human resources': 8,
    'company': 8, 'organization': 7, 'corporate': 7,
    # Business processes (weight 6-7)
    'procedure': 7, 'procedures': 7, 'guidelines': 6, 'handbook': 7,
    'benefits': 6, 'payroll': 6, 'leave': 5, 'vacation': 4,
    # Compliance and legal (weight 8-9)
    'compliance': 8, 'legal': 7, 'regulation': 7, 'audit': 6,
    # Training and development (weight 5-6)
    'training': 5, 'onboarding': 6, 'development': 4,
    # General inquiries (weight 3-4)
    'question': 3, 'inquiry': 3, 'help': 3, 'support': 4
}

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

@tool
def setup_email_monitoring(check_interval: int = 60) -> str:
    """Setup intelligent email monitoring to automatically read and respond to company-related emails.
    
    Args:
        check_interval: How often to check for new emails in seconds (default: 60)
    """
    global email_monitor_running
    
    # Get IMAP credentials
    imap_server = os.getenv("IMAP_SERVER")
    imap_user = os.getenv("IMAP_USER") or os.getenv("SMTP_USER")
    imap_password = os.getenv("IMAP_PASSWORD") or os.getenv("SMTP_PASSWORD")
    
    if not all([imap_server, imap_user, imap_password]):
        return "Missing IMAP configuration. Please set IMAP_SERVER, IMAP_USER, and IMAP_PASSWORD in your .env file."
    
    if email_monitor_running:
        return "Email monitoring is already running."
    
    # Start email monitoring in a separate thread
    def monitor_emails():
        global email_monitor_running, processed_emails
        email_monitor_running = True
        print(f"📧 Starting intelligent email monitoring on {imap_server}...")
        
        while email_monitor_running:
            try:
                # Connect to IMAP server
                mail = imaplib.IMAP4_SSL(imap_server, 993)
                mail.login(imap_user, imap_password)
                mail.select('INBOX')
                
                # Search for unread emails
                status, messages = mail.search(None, 'UNSEEN')
                
                if status == 'OK' and messages[0]:
                    email_ids = messages[0].split()
                    
                    for email_id in email_ids:
                        if email_id.decode() not in processed_emails:
                            try:
                                # Process and rank incoming email
                                process_incoming_email(mail, email_id)
                                processed_emails.add(email_id.decode())
                            except Exception as e:
                                print(f"❌ Error processing email {email_id}: {e}")
                
                mail.logout()
                
            except Exception as e:
                print(f"❌ Email monitoring error: {e}")
            
            # Wait before next check
            time.sleep(check_interval)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_emails, daemon=True)
    monitor_thread.start()
    
    return f"✅ Intelligent email monitoring started! Checking every {check_interval} seconds on {imap_server}\n🎯 Will auto-respond to company-related emails and rank by importance."

@tool
def stop_email_monitoring() -> str:
    """Stop the automatic email monitoring."""
    global email_monitor_running
    
    if not email_monitor_running:
        return "Email monitoring is not currently running."
    
    email_monitor_running = False
    return "✅ Email monitoring stopped."

@tool
def read_ranked_emails(count: int = 10) -> str:
    """Read and rank recent emails by importance, focusing on company-related content.
    
    Args:
        count: Number of recent emails to read and rank (default: 10)
    """
    try:
        # Get IMAP credentials
        imap_server = os.getenv("IMAP_SERVER")
        imap_user = os.getenv("IMAP_USER") or os.getenv("SMTP_USER")
        imap_password = os.getenv("IMAP_PASSWORD") or os.getenv("SMTP_PASSWORD")
        
        if not all([imap_server, imap_user, imap_password]):
            return "Missing IMAP configuration in .env file."
        
        # Connect to IMAP server
        mail = imaplib.IMAP4_SSL(imap_server, 993)
        mail.login(imap_user, imap_password)
        mail.select('INBOX')
        
        # Search for recent emails
        status, messages = mail.search(None, 'ALL')
        
        if status != 'OK':
            return "Failed to search emails."
        
        email_ids = messages[0].split()
        emails_with_scores = []
        
        # Get and score the most recent emails
        for email_id in email_ids[-count:]:
            try:
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                if status == 'OK':
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    # Extract email details
                    subject = decode_header(email_message["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    
                    sender = email_message["From"]
                    date = email_message["Date"]
                    
                    # Get email body
                    body = extract_email_body(email_message)
                    
                    # Calculate importance score
                    importance_score = calculate_importance_score(subject, body)
                    
                    emails_with_scores.append({
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'body': body[:200] + "..." if len(body) > 200 else body,
                        'importance': importance_score,
                        'is_company_related': importance_score > 3
                    })
                    
            except Exception as e:
                print(f"Error reading email {email_id}: {e}")
        
        mail.logout()
        
        if not emails_with_scores:
            return "No recent emails found."
        
        # Sort by importance score (highest first)
        emails_with_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        # Format the results
        result = f"📧 Found {len(emails_with_scores)} emails ranked by importance:\n\n"
        
        for i, email_info in enumerate(emails_with_scores, 1):
            importance_level = get_importance_level(email_info['importance'])
            company_indicator = "🏢 COMPANY" if email_info['is_company_related'] else "📝 General"
            
            result += f"{i}. {importance_level} {company_indicator} (Score: {email_info['importance']})\n"
            result += f"   FROM: {email_info['sender']}\n"
            result += f"   SUBJECT: {email_info['subject']}\n"
            result += f"   DATE: {email_info['date']}\n"
            result += f"   PREVIEW: {email_info['body']}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error reading emails: {str(e)}"

def calculate_importance_score(subject, body):
    """Calculate importance score based on keywords in subject and body."""
    global IMPORTANCE_KEYWORDS
    
    # Combine subject and body for analysis
    text = f"{subject} {body}".lower()
    
    score = 0
    matched_keywords = []
    
    # Check for importance keywords
    for keyword, weight in IMPORTANCE_KEYWORDS.items():
        if keyword in text:
            score += weight
            matched_keywords.append(keyword)
    
    # Bonus for multiple company-related keywords
    company_keywords = ['policy', 'policies', 'hr', 'human resources', 'company', 'corporate']
    company_matches = sum(1 for kw in company_keywords if kw in text)
    if company_matches > 1:
        score += company_matches * 2
    
    return min(score, 50)  # Cap at 50

def get_importance_level(score):
    """Convert numerical score to importance level."""
    if score >= 20:
        return "🔴 CRITICAL"
    elif score >= 15:
        return "🟠 HIGH"
    elif score >= 8:
        return "🟡 MEDIUM"
    elif score >= 4:
        return "🟢 LOW"
    else:
        return "⚪ MINIMAL"

def extract_email_body(email_message):
    """Extract the body text from an email message."""
    body = ""
    
    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            if content_type == "text/plain" and "attachment" not in content_disposition:
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                break
    else:
        body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
    
    return body.strip()

def should_auto_respond(subject, body):
    """Determine if email should receive an automatic response based on company relevance."""
    text = f"{subject} {body}".lower()
    
    # Company-related keywords that trigger auto-response
    trigger_keywords = [
        'policy', 'policies', 'hr', 'human resources', 'company',
        'procedure', 'procedures', 'handbook', 'benefits', 'payroll',
        'compliance', 'training', 'onboarding'
    ]
    
    return any(keyword in text for keyword in trigger_keywords)

def process_incoming_email(mail, email_id):
    """Process an incoming email with intelligent analysis and auto-response."""
    global document_content
    
    try:
        # Fetch email
        status, msg_data = mail.fetch(email_id, '(RFC822)')
        
        if status != 'OK':
            return
        
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        # Extract email details
        subject = decode_header(email_message["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        
        sender = email_message["From"]
        sender_email = parseaddr(sender)[1].lower()
        
        body = extract_email_body(email_message)
        
        # Calculate importance
        importance_score = calculate_importance_score(subject, body)
        importance_level = get_importance_level(importance_score)
        
        print(f"\n📨 New email received [{importance_level}]:")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Importance Score: {importance_score}")
        print(f"Body preview: {body[:100]}...")
        
        # Check if this is a company-related email that should receive auto-response
        if should_auto_respond(subject, body):
            print(f"🎯 Company-related email detected - generating intelligent response...")
            generate_intelligent_response(sender_email, subject, body)
        else:
            print(f"📝 General email - no auto-response needed")
        
    except Exception as e:
        print(f"❌ Error processing email {email_id}: {e}")

def generate_intelligent_response(sender_email, subject, body):
    """Generate an intelligent AI-powered response using React prompting."""
    global document_content
    
    try:
        # Search company database for relevant information
        search_query = f"{subject} {body[:200]}"
        db_result = search_company_db(search_query)
        
        # React prompting format for AI response generation
        react_prompt = f"""
You are an intelligent email assistant for our company's HR department. Use the React (Reason, Act, Observe) approach to generate a professional email response.

**THOUGHT**: I need to analyze this incoming email and provide a helpful response based on company information.

**REASONING**: 
- The email is from: {sender_email}
- Subject: {subject}
- Content shows they are asking about: {body[:300]}
- Available company information: {db_result}

**ACTION**: Generate a professional email response that:
1. Acknowledges their inquiry professionally
2. Provides relevant information from our company database
3. Offers additional assistance if needed
4. Maintains a helpful, professional tone
5. Uses proper email format with To:, Subject:, and body
6. Starts with "To: [sender_email]"...the specific recipient email/name
7. Ends with "Best regards, HR Team"

**OBSERVATION**: The response should be formatted as a complete email ready to send.

Generate the email response now:
"""
        
        model_temp = ChatOpenAI(model="gpt-4o")
        ai_response = model_temp.invoke([HumanMessage(content=react_prompt)])
        
        # Ensure proper email format
        response_content = ai_response.content
        
        if not response_content.startswith("To:"):
            response_content = f"To: {sender_email}\nSubject: Re: {subject}\n\n{response_content}"
        
        document_content = response_content
        
        # Send the AI-generated response
        send_result = send_email("")
        print(f"🤖 Intelligent response sent: {send_result}")
        
    except Exception as e:
        print(f"❌ Error generating intelligent response: {e}")

def check_email_configuration():
    """Check if email configuration is complete."""
    required_email_vars = ["IMAP_SERVER", "SMTP_SERVER", "SMTP_USER", "SMTP_PASSWORD"]
    email_vars = [var for var in required_email_vars if os.getenv(var)]
    
    if len(email_vars) >= 3:  # At least SMTP configured
        print("✅ Email configuration detected")
        return True
    else:
        print("⚠️ Email configuration incomplete - some email features may not work")
        print("💡 Add IMAP_SERVER, SMTP_SERVER, SMTP_USER, SMTP_PASSWORD to .env for full functionality")
        return False

# Updated tools list with email functionality
tools = [update, save, send_email, search_company_db, search_web, draft_from_db, 
         setup_email_monitoring, stop_email_monitoring, read_ranked_emails]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def get_user_input():
    """Get user input with better error handling"""
    try:
        user_input = input("\n🤔 What would you like to do? (draft/update/search/save/send/email/quit): ").strip()
        return user_input
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        return "quit"

def our_agent(state: AgentState) -> AgentState:     
    global document_content      
    system_prompt = SystemMessage(content=f"""           
# Enhanced Drafter AI Agent with Email Intelligence

You are Drafter, a professional writing assistant with advanced email monitoring and auto-response capabilities. You use React (Reason, Act, Observe) prompting for all decisions and responses.

## React Prompting Framework
For every user request, follow this structure:
**THOUGHT**: What is the user asking for?
**REASONING**: Why should I take this approach?
**ACTION**: What specific action will I take?
**OBSERVATION**: What was the result?

## Core Intelligence Rules

### 1. Document Type Recognition (React Approach)
**THOUGHT**: User wants me to create content
**REASONING**: Need to determine if this is email, memo, or document based on context clues
**ACTION**: Analyze for indicators:
- **Email Indicators**: "email to [name]", "send to [name]", "write [name]", mentions of recipients
- **Memo Indicators**: "memo", "internal memo", "company memo", "staff memo"
- **Other Documents**: "report", "letter", "notice", "announcement"
**OBSERVATION**: Choose appropriate format

### 2. Email Intelligence Features
- **Importance Ranking**: Automatically ranks emails by company relevance and urgency
- **Auto-Response**: Only responds to company-related emails (policies, HR, procedures, rules, regulations etc.)
- **Smart Monitoring**: Continuous background monitoring with intelligent filtering

### 3. Email Importance Scoring
**HIGH PRIORITY** (Auto-respond): policy, hr, company, compliance, procedures, benefits, training, rules, regulations, onboarding, legal, critical issues
**MEDIUM PRIORITY**: general business inquiries
**LOW PRIORITY**: personal or non-business emails

## Email Format Template (React-Generated)
```
To: [extracted_recipient_email]
Subject: [subject_from_context]

Dear [recipient_name]# extracted from email or context

[Body content - professional and helpful]

Best regards,
HR Team
```

## MANDATORY FORMATTING RULES
- **NEVER** use placeholder text like "[Your Name]", "[Company Name],"[Employee/Team/Department Name]"
- **ALWAYS** end with "Best regards, HR Team"
- **NO CONTACT INFO** at the end # just the HR Team signature
- **NO PLACEHOLDERS** anywhere in the final document
- Extract actual names from user input, default to "Team" if unclear

## Email Commands Available
- "start monitoring emails" - Begin intelligent email monitoring
- "stop monitoring emails" - Stop email monitoring  
- "read ranked emails" - Show recent emails ranked by importance
- "check email [number]" - Read specific email details

## Reactive Response Rules

### React Decision Process for Each Request:
**THOUGHT**: What does the user need?
**REASONING**: Based on keywords and context, what's the best approach?
**ACTION**: Use appropriate tool and format
**OBSERVATION**: Verify result meets requirements

### Length Guidelines
- **Brief email**: Maximum 50 words
- **Standard response**: 50-100 words
- **Detailed explanation**: 150 words if requested

## Error Prevention with React
**THOUGHT**: Is this request complete and clear?
**REASONING**: Do I have all necessary information?
**ACTION**: Either proceed or ask for clarification
**OBSERVATION**: Ensure high-quality output

## Example React Response Flow

**User**: "Start monitoring emails for company policies"
**THOUGHT**: User wants email monitoring for company-related content
**REASONING**: This requires setting up intelligent monitoring focused on policy-related emails
**ACTION**: Call setup_email_monitoring tool with company focus
**OBSERVATION**: Confirm monitoring is active and explain what it will do

**User**: "Read my important emails"  
**THOUGHT**: User wants to see prioritized email list
**REASONING**: Should use ranking system to show most important emails first
**ACTION**: Call read_ranked_emails to get scored and sorted email list
**OBSERVATION**: Present emails with importance levels and company relevance indicators

Remember: Always use React prompting structure, focus on company-related emails for auto-responses, and maintain professional communication standards.

The current document content is:\n{document_content}
    """)
    if not state["messages"]:
        user_input = "I'm ready to help you with documents and intelligent email management. What would you like to do?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do? ")
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
    
    print("\n🚀 Starting Monitor...")
    
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