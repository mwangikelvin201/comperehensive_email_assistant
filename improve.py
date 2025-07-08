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
# Email reading imports
import imaplib
import email
from email.header import decode_header
import time
from datetime import datetime, timedelta,timezone
import json
import threading
from email.utils import parseaddr
# Calendar imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import pickle
import dateutil.parser
from dateutil import tz

load_dotenv()

# Global variables
vectorstore = None
email_monitor_running = False
processed_emails = set()
calendar_service = None

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar', 
          'https://www.googleapis.com/auth/contacts.readonly']

# Simplified importance keywords
IMPORTANCE_KEYWORDS = {
    'urgent': 10, 'emergency': 10, 'critical': 10, 'asap': 10,
    'meeting': 9, 'appointment': 9, 'schedule': 8, 'calendar': 8,
    'policy': 9, 'hr': 8, 'company': 8, 'procedure': 7,
    'benefits': 6, 'training': 5, 'question': 3
}

# Simplified response templates
RESPONSE_TEMPLATES = {
    'company_query': """Dear {name},

Thank you for your inquiry. Based on our company policies and procedures, 

{db_content}

In the case of further enquiries, please don't hesitate to reach out.

Best regards,
HR Team""",
    
    'meeting_confirmation': """Dear {name},

Thank you for your meeting request. I'm pleased to confirm:

📅 Date: {date}
🕐 Time: {time}
⏱️ Duration: 1 hour

I've sent you a calendar invitation. Please let me know if you need to reschedule.

Best regards,
HR Team""",
    
    'meeting_conflict': """Dear {name},

Thank you for your meeting request. I have the following times available:

{available_times}

Please let me know which time works best for you.

Best regards,
HR Team"""
}

# Calendar keywords
CALENDAR_KEYWORDS = ['meeting', 'appointment', 'schedule', 'calendar', 'book', 'available', 'time']

def initialize_calendar_api():
    """Initialize Google Calendar API with better error handling."""
    global calendar_service
    
    try:
        print("🔧 Initializing Google Calendar API...")
        
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    print("⚠️ credentials.json not found. Calendar features disabled.")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        calendar_service = build('calendar', 'v3', credentials=creds)
        print("✅ Google Calendar API initialized successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Calendar API initialization failed: {e}")
        return False

def initialize_pinecone():
    """Initialize Pinecone vector store."""
    global vectorstore
    try:
        print("🔧 Initializing Pinecone...")
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("⚠️ PINECONE_API_KEY not found")
            return False
            
        pc = pinecone.Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "company-policies")
        
        try:
            indexes = pc.list_indexes()
            if index_name not in [index.name for index in indexes]:
                print(f"⚠️ Index '{index_name}' not found")
                return False
        except Exception as e:
            print(f"⚠️ Could not access Pinecone: {e}")
            return False
            
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        print(f"✅ Pinecone connected: {index_name}")
        return True
        
    except Exception as e:
        print(f"⚠️ Pinecone initialization failed: {e}")
        return False

# Initialize services
pinecone_available = initialize_pinecone()
calendar_available = initialize_calendar_api()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def search_company_db(query: str) -> str:
    """Search company database with better filtering."""
    global vectorstore
    
    if not vectorstore:
        return "Company database not available"
    
    try:
        print(f"🔍 Searching company database: {query}")
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant information found in company database"
        
        # Filter and combine relevant content
        relevant_content = []
        for doc in docs:
            content = doc.page_content.strip()
            if len(content) > 50:  # Filter out very short fragments
                relevant_content.append(content)
        
        if not relevant_content:
            return "No detailed information found in company database"
        
        # Combine and limit content
        combined_content = "\n\n".join(relevant_content[:1])  # Take top 2 results
        
        # Limit length
        if len(combined_content) > 800:
            combined_content = combined_content[:800] + "..."
        
        return combined_content
        
    except Exception as e:
        print(f"❌ Database search error: {e}")
        return "Error accessing company database"

def is_company_query(subject, body):
    """Check if email is a company policy/procedure query."""
    text = f"{subject} {body}".lower()
    company_keywords = ['policy', 'procedure', 'hr', 'company', 'benefits', 'handbook', 'guidelines','rules']
    return any(keyword in text for keyword in company_keywords)

def is_calendar_request(subject, body):
    """Check if email is a calendar/meeting request."""
    text = f"{subject} {body}".lower()
    return any(keyword in text for keyword in CALENDAR_KEYWORDS)

def extract_sender_name(sender_email):
    """Extract clean sender name."""
    name, email_addr = parseaddr(sender_email)
    if name:
        return name.replace('"', '').strip()
    else:
        username = email_addr.split('@')[0]
        return username.replace('.', ' ').replace('_', ' ').title()

def check_calendar_availability(days_ahead=7):
    """Check calendar availability for next few days."""
    global calendar_service
    
    if not calendar_service:
        return []
    
    try:
        from datetime import timezone
        now = datetime.now(timezone.utc)
        time_min = now.isoformat() 
        time_max = (now + timedelta(days=days_ahead)).isoformat() 
        
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Generate available time slots
        available_times = []
        for day in range(1, days_ahead + 1):
            check_date = now + timedelta(days=day)
            
            # Check common meeting times
            for hour in [9, 10, 11, 12, 14, 15, 16]:  # 9am-12am, 2pm-4pm
                slot_start = check_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                slot_end = slot_start + timedelta(hours=1)
                
                # Check if slot conflicts with existing events
                has_conflict = False
                for event in events:
                    event_start = event['start'].get('dateTime', event['start'].get('date'))
                    event_end = event['end'].get('dateTime', event['end'].get('date'))
                    
                    if 'T' in event_start:
                        event_start_dt = dateutil.parser.parse(event_start)
                        event_end_dt = dateutil.parser.parse(event_end)
                        
                        # Check for overlap
                        if (slot_start < event_end_dt and slot_end > event_start_dt):
                            has_conflict = True
                            break
                
                if not has_conflict:
                    available_times.append({
                        'datetime': slot_start,
                        'formatted': slot_start.strftime('%A, %B %d at %I:%M %p')
                    })
                    
                    if len(available_times) >= 3:  # Return first 3 available slots
                        break
            
            if len(available_times) >= 3:
                break
        
        return available_times
        
    except Exception as e:
        print(f"❌ Calendar availability check failed: {e}")
        return []

def create_calendar_event_simple(title, start_datetime, attendee_email):
    """Create a simple calendar event."""
    global calendar_service
    
    if not calendar_service:
        return False
    
    try:
        end_datetime = start_datetime + timedelta(hours=1)
        
        event = {
            'summary': title,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': 'UTC',
            },
            'attendees': [{'email': attendee_email}],
        }
        
        created_event = calendar_service.events().insert(
            calendarId='primary',
            body=event
        ).execute()
        
        print(f"✅ Calendar event created: {created_event.get('id')}")
        return True
        
    except Exception as e:
        print(f"❌ Calendar event creation failed: {e}")
        return False

def send_email_simple(to_email, subject, body):
    """Send email with simplified error handling."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        print("❌ SMTP configuration missing")
        return False

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())

        print(f"✅ Email sent to {to_email}")
        return True

    except Exception as e:
        print(f"❌ Email sending failed: {e}")
        return False

def process_email_smart(mail, email_id):
    """Process email with improved logic."""
    try:
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
        sender_name = extract_sender_name(sender)
        
        # Extract body
        body = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        body = body.strip()
        
        print(f"\n📨 Processing email:")
        print(f"From: {sender_name} <{sender_email}>")
        print(f"Subject: {subject}")
        
        # Determine email type and handle accordingly
        if is_calendar_request(subject, body):
            print("📅 Detected: Calendar/Meeting request")
            handle_calendar_request(sender_email, sender_name, subject, body)
            
        elif is_company_query(subject, body):
            print("🏢 Detected: Company query")
            handle_company_query(sender_email, sender_name, subject, body)
            
        else:
            print("📝 Detected: General email (no auto-response)")
        
    except Exception as e:
        print(f"❌ Email processing error: {e}")

def handle_calendar_request(sender_email, sender_name, subject, body):
    """Handle calendar/meeting requests efficiently."""
    print("🔄 Processing calendar request...")
    
    # Check available times
    available_times = check_calendar_availability()
    
    if not available_times:
        # No calendar service or no available times
        response_body = f"""Dear {sender_name},

Thank you for your meeting request. I'm currently reviewing my schedule and will get back to you within 24 hours with available time slots.

Best regards,
HR Team"""
        
        send_email_simple(sender_email, f"Re: {subject}", response_body)
        return
    
    # Get first available time
    first_available = available_times[0]
    
    # Create calendar event
    event_created = create_calendar_event_simple(
        f"Meeting with {sender_name}",
        first_available['datetime'],
        sender_email
    )
    
    if event_created:
        # Send confirmation
        response_body = RESPONSE_TEMPLATES['meeting_confirmation'].format(
            name=sender_name,
            date=first_available['datetime'].strftime('%A, %B %d, %Y'),
            time=first_available['datetime'].strftime('%I:%M %p')
        )
        
        send_email_simple(sender_email, f"Meeting Confirmed: {subject}", response_body)
        
    else:
        # Send available times
        times_text = "\n".join([f"• {slot['formatted']}" for slot in available_times[:3]])
        
        response_body = RESPONSE_TEMPLATES['meeting_conflict'].format(
            name=sender_name,
            available_times=times_text
        )
        
        send_email_simple(sender_email, f"Re: {subject}", response_body)

def handle_company_query(sender_email, sender_name, subject, body):
    """Handle company policy/procedure queries."""
    print("🔄 Processing company query...")
    
    # Search company database
    search_query = f"{subject} {body}"
    db_content = search_company_db(search_query)
    
    # Generate response
    response_body = RESPONSE_TEMPLATES['company_query'].format(
        name=sender_name,
        db_content=db_content
    )
    
    send_email_simple(sender_email, f"Re: {subject}", response_body)

@tool
def start_email_monitoring(check_interval: int = 60) -> str:
    """Start smart email monitoring."""
    global email_monitor_running
    
    # Check configuration
    required_vars = ["IMAP_SERVER", "IMAP_USER", "IMAP_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        return f"❌ Missing IMAP configuration: {', '.join(missing_vars)}"
    
    if email_monitor_running:
        return "✅ Email monitoring already running"
    
    def monitor_emails():
        global email_monitor_running, processed_emails
        email_monitor_running = True
        
        imap_server = os.getenv("IMAP_SERVER")
        imap_user = os.getenv("IMAP_USER")
        imap_password = os.getenv("IMAP_PASSWORD")
        
        print(f"📧 Starting email monitoring on {imap_server}...")
        
        while email_monitor_running:
            try:
                mail = imaplib.IMAP4_SSL(imap_server, 993)
                mail.login(imap_user, imap_password)
                mail.select('INBOX')
                
                status, messages = mail.search(None, 'UNSEEN')
                
                if status == 'OK' and messages[0]:
                    email_ids = messages[0].split()
                    print(f"📬 Found {len(email_ids)} new emails")
                    
                    for email_id in email_ids:
                        if email_id.decode() not in processed_emails:
                            process_email_smart(mail, email_id)
                            processed_emails.add(email_id.decode())
                
                mail.logout()
                
            except Exception as e:
                print(f"❌ Email monitoring error: {e}")
            
            time.sleep(check_interval)
    
    monitor_thread = threading.Thread(target=monitor_emails, daemon=True)
    monitor_thread.start()
    
    features = []
    if pinecone_available:
        features.append("Company DB queries")
    if calendar_available:
        features.append("Calendar booking")
    
    return f"✅ Email monitoring started!\n🔧 Features: {', '.join(features)}\n⏱️ Check interval: {check_interval}s"

@tool
def stop_email_monitoring() -> str:
    """Stop email monitoring."""
    global email_monitor_running
    
    if not email_monitor_running:
        return "❌ Email monitoring not running"
    
    email_monitor_running = False
    return "✅ Email monitoring stopped"

@tool
def check_email_status() -> str:
    """Check current email monitoring status."""
    status = "✅ Running" if email_monitor_running else "❌ Stopped"
    
    features = []
    if pinecone_available:
        features.append("✅ Company DB")
    else:
        features.append("❌ Company DB")
        
    if calendar_available:
        features.append("✅ Calendar")
    else:
        features.append("❌ Calendar")
    
    return f"📊 Email Monitor Status: {status}\n🔧 Features: {', '.join(features)}\n📈 Processed emails: {len(processed_emails)}"

# Tools list
tools = [start_email_monitoring, stop_email_monitoring, check_email_status]

# Model
model = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(tools)

def email_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
# Smart Email Assistant

You are an intelligent email assistant that automatically handles:

## 📧 Email Types:
1. **Company Queries** - Policy, procedure, HR, benefits questions
   - Search company database for relevant information
   - Provide comprehensive, helpful responses
   
2. **Calendar Requests** - Meeting, appointment, scheduling requests
   - Check calendar availability
   - Book meetings automatically
   - Send confirmations

## 🎯 Core Features:
- **Auto-categorization** of incoming emails
- **Smart responses** based on email type
- **Calendar integration** for seamless booking
- **Database search** for company information

## 📋 Available Commands:
- `start monitoring` - Begin email monitoring
- `stop monitoring` - Stop email monitoring  
- `check status` - View monitoring status

## 🔧 Smart Processing:
- Automatically detects email type
- Searches company database for policy questions
- Books calendar appointments for meeting requests
- Sends professional, personalized responses

Ready to assist with intelligent email management!
""")
    
    if not state["messages"]:
        user_input = "Smart Email Assistant ready! Type 'start monitoring' to begin."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\n🎯 Command (start monitoring/stop monitoring/check status/quit): ")
        print(f"👤 USER: {user_input}")
        
        if user_input.strip().lower() == "quit":
            return {"messages": list(state["messages"]) + [HumanMessage(content="quit")]}
        
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"🤖 ASSISTANT: {response.content}")
    
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 EXECUTING: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Check if we should continue."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content.lower().strip() == "quit":
            return "end"
        
    return "continue"

# Graph setup
graph = StateGraph(AgentState)
graph.add_node("agent", email_agent)
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

app = graph.compile()

def run_email_assistant():
    """Run the email assistant."""
    print("\n" + "="*60)
    print("🚀 SMART EMAIL ASSISTANT")
    print("="*60)
    print("📧 Automatic Email Processing")
    print("📅 Calendar Integration")
    print("🏢 Company Database Search")
    print("="*60)
    
    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found!")
        return
    
    # Show status
    print(f"📊 Company DB: {'✅ Connected' if pinecone_available else '❌ Not available'}")
    print(f"📅 Calendar: {'✅ Connected' if calendar_available else '❌ Not available'}")
    
    try:
        state = {"messages": []}
        
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                messages = step["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, ToolMessage):
                        print(f"📋 RESULT: {last_message.content}")
                
    except KeyboardInterrupt:
        print("\n👋 Email assistant stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_email_assistant()