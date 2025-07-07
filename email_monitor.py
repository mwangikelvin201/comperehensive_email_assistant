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
from datetime import datetime, timedelta
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
from langchain.chains.summarize import load_summarize_chain


load_dotenv()

# Global variables
vectorstore = None
email_monitor_running = False
processed_emails = set()
response_cache = {}  # Cache for similar queries to reduce API calls
calendar_service = None
contacts_service = None

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar', 
          'https://www.googleapis.com/auth/contacts.readonly']

# Email importance keywords with weights
IMPORTANCE_KEYWORDS = {
    # High importance (weight 10)
    'urgent': 10, 'emergency': 10, 'critical': 10, 'asap': 10,
    # Calendar/Meeting related (weight 8-9)
    'meeting': 9, 'appointment': 9, 'schedule': 8, 'calendar': 8,
    'book': 7, 'booking': 7, 'available': 6, 'reschedule': 7,
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

# Enhanced response templates with better personalization
RESPONSE_TEMPLATES = {
    'policy': """Dear {name},

I hope this message finds you well.

Thank you for your inquiry about our company policies. I've reviewed your request and found the following information:

{db_content}

If you need further clarification on any specific aspect of our policies, please don't hesitate to reach out. I'm here to help ensure you have all the information you need.

Best regards,
HR Team""",
    
    'benefits': """Dear {name},

I hope you're doing well.

Thank you for your question regarding employee benefits. Here's what I found that should address your inquiry:

{db_content}

For personalized details about your specific benefits package or to discuss any concerns, please feel free to contact me directly. I'm happy to schedule a brief call if that would be helpful.

Best regards,
HR Team""",
    
    'procedure': """Dear {name},

I hope this message finds you well.

Thank you for your inquiry about our procedures. I've gathered the relevant information for you:

{db_content}

If you need any clarification on the steps involved or have questions about implementation, please let me know. I'm here to support you through the process.

Best regards,
HR Team""",
    
    'training': """Dear {name},

I hope you're having a great day.

Thank you for your interest in training opportunities. Here's what I found regarding your inquiry:

{db_content}

I'll keep you informed about upcoming training sessions that align with your interests. Please let me know if you'd like to discuss specific training needs or career development goals.

Best regards,
HR Team""",
    
    'meeting': """Dear {name},

I hope this message finds you well.

Thank you for your meeting request. I've reviewed your availability and the details you provided:

{db_content}

I'll check our calendar and get back to you with some suitable time slots. If you have any specific preferences or additional requirements for the meeting, please let me know.

Looking forward to our discussion.

Best regards,
HR Team""",
    
    'follow_up': """Dear {name},

I hope this message finds you well.

I am writing to follow up on {subject} that we discussed. As we move forward with our timeline, I wanted to check in on the progress and see if there are any updates or challenges you'd like to discuss.

{db_content}

Could you please share a brief summary or let me know if you'd like to set up a short call this week? This will help us align our next steps with the broader strategy.

Thank you for your continued efforts and collaboration. I look forward to hearing from you.

Warm regards,
HR Team""",
    
    'general': """Dear {name},

I hope this message finds you well.

Thank you for reaching out. I've reviewed your message and here's what I found:

{db_content}

If you have any further questions or need additional assistance, please feel free to contact me. I'm here to help and ensure you have the support you need.

Best regards,
HR Team"""
}

# Calendar and booking keywords
CALENDAR_KEYWORDS = [
    'meeting', 'appointment', 'schedule', 'calendar', 'book', 'booking',
    'available', 'reschedule', 'call', 'conference', 'zoom', 'teams',
    'discuss', 'review', 'presentation', 'demo', 'interview'
]

BOOKING_KEYWORDS = [
    'book', 'booking', 'schedule', 'appointment', 'meeting', 'available',
    'time slot', 'calendar', 'free time', 'when are you available'
]

def summarize_db_content(content):
    chain = load_summarize_chain(model, chain_type="stuff")
    return chain.run(content)

# Initialize Google Calendar API
def initialize_calendar_api():
    """Initialize Google Calendar API."""
    global calendar_service, contacts_service
    
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
                    print("⚠️ credentials.json not found. Please download from Google Cloud Console.")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
        
        calendar_service = build('calendar', 'v3', credentials=creds)
        contacts_service = build('people', 'v1', credentials=creds)
        
        print("✅ Google Calendar API initialized successfully")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not initialize Google Calendar API: {e}")
        print("💡 Make sure you have credentials.json in the project directory")
        return False

# Initialize Pinecone with better error handling
def initialize_pinecone():
    global vectorstore
    try:
        print("🔧 Initializing Pinecone...")
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("⚠️ PINECONE_API_KEY not found in .env file")
            return False
            
        pc = pinecone.Pinecone(api_key=api_key)
        index_name = os.getenv("PINECONE_INDEX_NAME", "company-policies")
        
        print(f"🔍 Checking for index: {index_name}")
        
        try:
            indexes = pc.list_indexes()
            if index_name not in [index.name for index in indexes]:
                print(f"⚠️ Warning: Pinecone index '{index_name}' not found. Will use templates only.")
                return False
        except Exception as e:
            print(f"⚠️ Could not list indexes: {e}")
            return False
            
        print("🔧 Initializing OpenAI embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
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
calendar_available = initialize_calendar_api()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def get_cached_response(query_key):
    """Get cached response to avoid API calls"""
    return response_cache.get(query_key)

def cache_response(query_key, response):
    """Cache response for future use"""
    response_cache[query_key] = response
    # Keep cache size manageable
    if len(response_cache) > 100:
        # Remove oldest entries
        keys_to_remove = list(response_cache.keys())[:20]
        for key in keys_to_remove:
            del response_cache[key]

def search_company_db(query: str) -> str:
    """Search the company's policy database for relevant content."""
    global vectorstore
    
    if not vectorstore:
        return "Database not available"
    
    try:
        # Check cache first
        query_key = query.lower().strip()
        cached_result = get_cached_response(query_key)
        if cached_result:
            print(f"📋 Using cached result for: {query}")
            return cached_result
        
        print(f"🔍 Searching company database for: {query}")
        docs = vectorstore.similarity_search(query, k=2)  # Reduced from 3 to 2
        
        if not docs:
            result = f"No relevant information found for: {query}"
        else:
            content = []
            for i, doc in enumerate(docs, 1):
                content.append(f"{doc.page_content}")
            result = "\n\n".join(content)
        
        # Cache the result
        cache_response(query_key, result)
        return result
        
    except Exception as e:
        print(f"❌ Error searching company database: {e}")
        return f"Error searching database: {str(e)}"

def determine_email_category(subject, body):
    """Determine email category to use appropriate template"""
    text = f"{subject} {body}".lower()
    
    # Check for calendar/meeting related content first
    if any(word in text for word in CALENDAR_KEYWORDS):
        return 'meeting'
    elif any(word in text for word in ['policy', 'policies']):
        return 'policy'
    elif any(word in text for word in ['benefit', 'benefits', 'insurance', 'health']):
        return 'benefits'
    elif any(word in text for word in ['procedure', 'procedures', 'process', 'guideline']):
        return 'procedure'
    elif any(word in text for word in ['training', 'course', 'learning', 'development']):
        return 'training'
    elif any(word in text for word in ['follow up', 'follow-up', 'followup', 'update', 'progress']):
        return 'follow_up'
    else:
        return 'general'

def extract_sender_name(sender_email):
    """Extract name from sender email"""
    name, email_addr = parseaddr(sender_email)
    if name:
        return name.replace('"', '').strip()
    else:
        # Extract name from email address
        username = email_addr.split('@')[0]
        return username.replace('.', ' ').replace('_', ' ').title()

def is_calendar_related(subject, body):
    """Check if email is calendar/meeting related"""
    text = f"{subject} {body}".lower()
    return any(keyword in text for keyword in CALENDAR_KEYWORDS)

def is_booking_request(subject, body):
    """Check if email is a booking request"""
    text = f"{subject} {body}".lower()
    return any(keyword in text for keyword in BOOKING_KEYWORDS)

def extract_meeting_details(subject, body):
    """Extract meeting details from email content"""
    text = f"{subject} {body}"
    
    # Simple extraction patterns - can be enhanced with NLP
    meeting_details = {
        'title': subject,
        'description': body[:200],
        'duration': 60,  # default 1 hour
        'attendees': []
    }
    
    # Extract time patterns (basic implementation)
    time_patterns = [
        r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
        r'(\d{1,2}\s*(?:AM|PM|am|pm))',
        r'at\s+(\d{1,2}:\d{2})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        r'(next\s+week|tomorrow|today)'
    ]
    
    for pattern in time_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            meeting_details['time_mention'] = matches[0]
            break
    
    return meeting_details

@tool
def check_calendar_events(days_ahead: int = 7) -> str:
    """Check upcoming calendar events for the next specified days."""
    global calendar_service
    
    if not calendar_service:
        return "Calendar service not available. Please check Google Calendar API setup."
    
    try:
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=days_ahead)).isoformat() + 'Z'
        
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        if not events:
            return f"No upcoming events found for the next {days_ahead} days."
        
        result = f"📅 Upcoming events for the next {days_ahead} days:\n\n"
        
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            summary = event.get('summary', 'No title')
            description = event.get('description', '')
            attendees = event.get('attendees', [])
            
            # Parse datetime
            if 'T' in start:
                start_dt = dateutil.parser.parse(start)
                start_str = start_dt.strftime('%Y-%m-%d %H:%M')
            else:
                start_str = start
            
            result += f"🗓️ {summary}\n"
            result += f"   📍 Time: {start_str}\n"
            
            if attendees:
                attendee_list = [a.get('email', '') for a in attendees]
                result += f"   👥 Attendees: {', '.join(attendee_list)}\n"
            
            if description:
                result += f"   📝 Description: {description[:100]}...\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error checking calendar: {str(e)}"

@tool
def create_calendar_event(title: str, start_time: str, end_time: str, description: str = "", attendees: list = None) -> str:
    """Create a new calendar event."""
    global calendar_service
    
    if not calendar_service:
        return "Calendar service not available. Please check Google Calendar API setup."
    
    try:
        event = {
            'summary': title,
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': 'UTC',
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'UTC',
            },
        }
        
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]
        
        created_event = calendar_service.events().insert(
            calendarId='primary',
            body=event
        ).execute()
        
        return f"✅ Calendar event created successfully: {created_event.get('htmlLink')}"
        
    except Exception as e:
        return f"Error creating calendar event: {str(e)}"

@tool
def send_follow_up_emails(days_ahead: int = 7) -> str:
    """Send follow-up emails for upcoming events."""
    global calendar_service
    
    if not calendar_service:
        return "Calendar service not available."
    
    try:
        # Get upcoming events
        events_response = check_calendar_events(days_ahead)
        
        if "No upcoming events" in events_response:
            return "No upcoming events found to send follow-ups for."
        
        follow_ups_sent = 0
        
        # Parse events and send follow-ups (simplified implementation)
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=datetime.utcnow().isoformat() + 'Z',
            timeMax=(datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        for event in events:
            attendees = event.get('attendees', [])
            summary = event.get('summary', 'Upcoming Meeting')
            
            for attendee in attendees:
                if attendee.get('email'):
                    # Send follow-up email
                    subject = f"Follow-up: {summary}"
                    body = f"""Dear {attendee.get('displayName', 'Colleague')},

I hope this message finds you well.

I wanted to follow up on our upcoming meeting: {summary}

Please let me know if you have any questions or if there are any materials you'd like to review beforehand.

Looking forward to our discussion.

Best regards,
HR Team"""
                    
                    send_result = send_email_directly(attendee['email'], subject, body)
                    if "successfully" in send_result:
                        follow_ups_sent += 1
        
        return f"✅ Sent {follow_ups_sent} follow-up emails for upcoming events."
        
    except Exception as e:
        return f"Error sending follow-up emails: {str(e)}"
    
def check_availability_direct(start_time: str, end_time: str) -> str:
    global calendar_service
    if not calendar_service:
        return "Calendar service not available."
    
    try:
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True
        ).execute()

        events = events_result.get('items', [])
        if not events:
            return f"✅ Time slot available: {start_time} to {end_time}"
        else:
            return f"❌ Time slot conflicts with {len(events)} existing event(s)"

    except Exception as e:
        return f"Error checking availability: {str(e)}"





@tool
def check_time_slot_availability(start_time: str, end_time: str) -> str:
    """Check if a time slot is available in the calendar (LangChain tool)."""
    return check_availability_direct(start_time, end_time)


@tool
def setup_email_monitoring(check_interval: int = 60) -> str:
    """Setup intelligent email monitoring with optimized API usage."""
    global email_monitor_running
    
    imap_server = os.getenv("IMAP_SERVER")
    imap_user = os.getenv("IMAP_USER") or os.getenv("SMTP_USER")
    imap_password = os.getenv("IMAP_PASSWORD") or os.getenv("SMTP_PASSWORD")
    
    if not all([imap_server, imap_user, imap_password]):
        return "Missing IMAP configuration. Please set IMAP_SERVER, IMAP_USER, and IMAP_PASSWORD in your .env file."
    
    if email_monitor_running:
        return "Email monitoring is already running."
    
    def monitor_emails():
        global email_monitor_running, processed_emails
        email_monitor_running = True
        print(f"📧 Starting enhanced email monitoring on {imap_server}...")
        
        while email_monitor_running:
            try:
                mail = imaplib.IMAP4_SSL(imap_server, 993)
                mail.login(imap_user, imap_password)
                mail.select('INBOX')
                
                status, messages = mail.search(None, 'UNSEEN')
                
                if status == 'OK' and messages[0]:
                    email_ids = messages[0].split()
                    
                    for email_id in email_ids:
                        if email_id.decode() not in processed_emails:
                            try:
                                process_incoming_email_enhanced(mail, email_id)
                                processed_emails.add(email_id.decode())
                            except Exception as e:
                                print(f"❌ Error processing email {email_id}: {e}")
                
                mail.logout()
                
            except Exception as e:
                print(f"❌ Email monitoring error: {e}")
            
            time.sleep(check_interval)
    
    monitor_thread = threading.Thread(target=monitor_emails, daemon=True)
    monitor_thread.start()
    
    return f"✅ Enhanced email monitoring started! Features include:\n- Calendar integration\n- Booking detection\n- Follow-up automation\n- Checking every {check_interval} seconds on {imap_server}"

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
    """Read and rank recent emails by importance."""
    try:
        imap_server = os.getenv("IMAP_SERVER")
        imap_user = os.getenv("IMAP_USER") or os.getenv("SMTP_USER")
        imap_password = os.getenv("IMAP_PASSWORD") or os.getenv("SMTP_PASSWORD")
        
        if not all([imap_server, imap_user, imap_password]):
            return "Missing IMAP configuration in .env file."
        
        mail = imaplib.IMAP4_SSL(imap_server, 993)
        mail.login(imap_user, imap_password)
        mail.select('INBOX')
        
        status, messages = mail.search(None, 'ALL')
        
        if status != 'OK':
            return "Failed to search emails."
        
        email_ids = messages[0].split()
        emails_with_scores = []
        
        for email_id in email_ids[-count:]:
            try:
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                if status == 'OK':
                    email_body = msg_data[0][1]
                    email_message = email.message_from_bytes(email_body)
                    
                    subject = decode_header(email_message["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode()
                    
                    sender = email_message["From"]
                    date = email_message["Date"]
                    
                    body = extract_email_body(email_message)
                    importance_score = calculate_importance_score(subject, body)
                    
                    emails_with_scores.append({
                        'id': email_id.decode(),
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'body': body[:200] + "..." if len(body) > 200 else body,
                        'importance': importance_score,
                        'is_company_related': importance_score > 3,
                        'is_calendar_related': is_calendar_related(subject, body),
                        'is_booking_request': is_booking_request(subject, body)
                    })
                    
            except Exception as e:
                print(f"Error reading email {email_id}: {e}")
        
        mail.logout()
        
        if not emails_with_scores:
            return "No recent emails found."
        
        emails_with_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        result = f"📧 Found {len(emails_with_scores)} emails ranked by importance:\n\n"
        
        for i, email_info in enumerate(emails_with_scores, 1):
            importance_level = get_importance_level(email_info['importance'])
            
            indicators = []
            if email_info['is_company_related']:
                indicators.append("🏢 COMPANY")
            if email_info['is_calendar_related']:
                indicators.append("📅 CALENDAR")
            if email_info['is_booking_request']:
                indicators.append("📋 BOOKING")
            if not indicators:
                indicators.append("📝 General")
            
            result += f"{i}. {importance_level} {' '.join(indicators)} (Score: {email_info['importance']})\n"
            result += f"   FROM: {email_info['sender']}\n"
            result += f"   SUBJECT: {email_info['subject']}\n"
            result += f"   DATE: {email_info['date']}\n"
            result += f"   PREVIEW: {email_info['body']}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error reading emails: {str(e)}"

def calculate_importance_score(subject, body):
    """Calculate importance score based on keywords."""
    global IMPORTANCE_KEYWORDS
    
    text = f"{subject} {body}".lower()
    score = 0
    
    for keyword, weight in IMPORTANCE_KEYWORDS.items():
        if keyword in text:
            score += weight
    
    # Bonus for multiple company-related keywords
    company_keywords = ['policy', 'policies', 'hr', 'human resources', 'company', 'corporate', 'procedure', 'procedures', 'guidelines', 'handbook', 'benefits', 'payroll']
    company_matches = sum(1 for kw in company_keywords if kw in text)
    if company_matches > 1:
        score += company_matches * 2
    
    # Bonus for calendar-related keywords
    calendar_matches = sum(1 for kw in CALENDAR_KEYWORDS if kw in text)
    if calendar_matches > 0:
        score += calendar_matches * 3
    
    return min(score, 50)

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
    """Determine if email should receive an automatic response."""
    text = f"{subject} {body}".lower()
    
    trigger_keywords = [
        'policy', 'policies', 'hr', 'human resources', 'company',
        'procedure', 'procedures', 'handbook', 'benefits', 'payroll',
        'compliance', 'training', 'onboarding', 'guidelines',
        'meeting', 'appointment', 'schedule', 'follow up', 'update'
    ]
    
    return any(keyword in text for keyword in trigger_keywords)

def send_email_directly(to_email, subject, body):
    """Send email directly without using AI tools."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        return "Missing SMTP configuration in .env file."

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())

        return f"Email sent successfully to {to_email}"

    except Exception as e:
        return f"Error sending email: {str(e)}"

def process_incoming_email_enhanced(mail, email_id):
    """Process incoming email with enhanced calendar integration and better responses."""
    try:
        status, msg_data = mail.fetch(email_id, '(RFC822)')
        
        if status != 'OK':
            return
        
        email_body = msg_data[0][1]
        email_message = email.message_from_bytes(email_body)
        
        subject = decode_header(email_message["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        
        sender = email_message["From"]
        sender_email = parseaddr(sender)[1].lower()
        sender_name = extract_sender_name(sender)
        
        body = extract_email_body(email_message)
        
        importance_score = calculate_importance_score(subject, body)
        importance_level = get_importance_level(importance_score)
        
        is_calendar = is_calendar_related(subject, body)
        is_booking = is_booking_request(subject, body)
        
        print(f"\n📨 New email received [{importance_level}]:")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Importance Score: {importance_score}")
        print(f"Calendar Related: {is_calendar}")
        print(f"Booking Request: {is_booking}")
        
        # Handle booking requests first
        if is_booking and calendar_service:
            print(f"📋 Processing booking request...")
            handle_booking_request(sender_email, sender_name, subject, body)
            
        elif should_auto_respond(subject, body):
            print(f"🎯 Generating enhanced response...")
            
            # Determine category and use appropriate template
            category = determine_email_category(subject, body)
            template = RESPONSE_TEMPLATES[category]
            
            # For calendar-related emails, don't search company DB
            if is_calendar:
                db_content = "I'll review this request and coordinate with the relevant team members to find the best time for our discussion."
            else:
                # Search database for relevant content only for non-calendar emails
                search_query = f"{subject} {body[:100]}"
                db_content = search_company_db(search_query)
                
                # Clean up database content to avoid policy fragments
      # Improved content handling from company database
            if "No relevant information found" in db_content or "Database not available" in db_content:
                print("⚠️ No useful DB content found, falling back to generic template.")
                if category == 'follow_up':
                    db_content = "I wanted to check in on our recent discussion and see how things are progressing on your end."
                else:
                    db_content = "We're currently reviewing this matter and will provide a comprehensive response shortly."
            elif len(db_content) > 1000:
                print("✂️ DB content too long, summarizing...")
                try:
                    db_content = summarize_db_content(db_content)
                except Exception as e:
                    print(f"⚠️ Summarization failed: {e}")
                    db_content = db_content[:1000] + "\n\n[Response trimmed for clarity]"
            else:
                print("✅ Using DB content directly in email response.")


            # Generate response using enhanced template
            if category == 'follow_up':
                response_body = template.format(
                    name=sender_name,
                    subject=subject.replace('Re: ', '').replace('RE: ', ''),
                    db_content=db_content
                )
            else:
                response_body = template.format(
                    name=sender_name,
                    db_content=db_content
                )
            
            response_subject = f"Re: {subject}"
            
            # Send response
            send_result = send_email_directly(sender_email, response_subject, response_body)
            print(f"📤 Enhanced response sent: {send_result}")
            
        else:
            print(f"📝 Email processed - no auto-response needed")
        
    except Exception as e:
        print(f"❌ Error processing email {email_id}: {e}")

def handle_booking_request(sender_email, sender_name, subject, body):
    """Handle booking requests by extracting details and creating calendar events."""
    global calendar_service
    
    try:
        print(f"🔍 Extracting meeting details from booking request...")
        
        # Extract meeting details
        meeting_details = extract_meeting_details(subject, body)
        
        # For now, suggest a default time slot (this can be enhanced with NLP)
        # In a real implementation, you'd parse the requested time from the email
        suggested_time = datetime.now() + timedelta(days=1)
        suggested_time = suggested_time.replace(hour=14, minute=0, second=0, microsecond=0)
        
        start_time = suggested_time.isoformat()
        end_time = (suggested_time + timedelta(hours=1)).isoformat()
        
        # Check availability
        availability_result = check_availability_direct(start_time, end_time)

        
        if "available" in availability_result:
            # Create calendar event
            event_result = create_calendar_event(
                title=f"Meeting with {sender_name}",
                start_time=start_time,
                end_time=end_time,
                description=f"Meeting requested via email. Original subject: {subject}",
                attendees=[sender_email]
            )
            
            # Send confirmation email
            confirmation_subject = f"Meeting Confirmed: {subject}"
            confirmation_body = f"""Dear {sender_name},

Thank you for your meeting request. I'm pleased to confirm that I've scheduled our meeting for:

📅 Date: {suggested_time.strftime('%A, %B %d, %Y')}
🕐 Time: {suggested_time.strftime('%I:%M %p')}
⏱️ Duration: 1 hour

I've sent you a calendar invitation with all the details. Please let me know if this time works for you or if you need to reschedule.

Looking forward to our discussion.

Best regards,
HR Team"""
            
            send_result = send_email_directly(sender_email, confirmation_subject, confirmation_body)
            print(f"📅 Meeting scheduled and confirmation sent: {send_result}")
            
        else:
            # Time slot not available, suggest alternative
            alt_subject = f"Re: {subject} - Alternative Time Needed"
            alt_body = f"""Dear {sender_name},

Thank you for your meeting request. Unfortunately, the requested time slot has a conflict with existing commitments.

I'd be happy to find an alternative time that works for both of us. Could you please let me know your availability for the following options:

• Tomorrow afternoon (2:00 PM - 4:00 PM)
• Day after tomorrow morning (10:00 AM - 12:00 PM)
• Any other time that works better for you

I'll send you a calendar invitation once we confirm the time.

Best regards,
HR Team"""
            
            send_result = send_email_directly(sender_email, alt_subject, alt_body)
            print(f"📅 Alternative time suggested: {send_result}")
            
    except Exception as e:
        print(f"❌ Error handling booking request: {e}")

# Enhanced tools list with calendar functions
tools = [
    setup_email_monitoring, 
    stop_email_monitoring, 
    read_ranked_emails,
    check_calendar_events,
    create_calendar_event,
    send_follow_up_emails,
    check_time_slot_availability
]

# Use a smaller, more efficient model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1).bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:     
    system_prompt = SystemMessage(content=f"""           
# Enhanced Autonomous Email Assistant

You are an intelligent, reactive email assistant that seamlessly integrates email management with calendar operations. Your core mission is to provide personalized, professional, and contextually appropriate responses while maintaining efficiency.

## Core Capabilities:

### 📧 Email Management
- Monitor incoming emails with intelligent prioritization
- Generate personalized, professional responses using appropriate templates
- Distinguish between company policy queries, scheduling and general communication
- Avoid robotic or template-heavy language in responses

### 📅 Calendar Integration  
- Check upcoming calendar events and identify stakeholders
- Automatically handle booking requests and schedule meetings
- Send proactive follow-up emails for upcoming events
- Verify time slot availability before scheduling
- Ensure the booked time is between 9 AM and 5 PM local time 

### 🎯 Response Strategy
- **FOR CALENDAR/MEETING EMAILS**: Focus on scheduling, availability, and coordination
- **FOR COMPANY POLICY EMAILS**: Always use database search when truly relevant 
- **FOR FOLLOW-UP EMAILS**: Maintain warm, professional tone with genuine interest
- **FOR GENERAL EMAILS**: Provide helpful, personalized responses

## Response Quality Standards:
- Always address recipients by name with proper salutation
- Use warm, professional tone ("I hope this message finds you well")
- Provide specific, actionable information when possible  
- Avoid lengthy database excerpts that don't directly answer the question
- End with genuine offers for further assistance
- Sign off appropriately as "HR Team" or similar

## Reactive Behavior:
- Continuously monitor for new emails and respond appropriately
- Automatically detect booking requests and process them
- Proactively send follow-ups for upcoming calendar events
- Adapt response style based on email content and context

## Available Commands:
- "start monitoring" - Begin enhanced email monitoring with calendar integration
- "stop monitoring" - Stop email monitoring
- "read emails" - Show recent emails with calendar/booking indicators
- "check calendar" - View upcoming events  
- "send follow-ups" - Send proactive follow-up emails for upcoming events

## Critical Rules:
- NEVER use generic policy fragments in responses
- ALWAYS personalize responses with recipient's name
- Always search company database when specifically relevant to the query
- AUTOMATICALLY handle booking requests by creating calendar events
- MAINTAIN professional yet warm communication style
- REACT promptly to calendar-related emails with appropriate actions

Remember: You are not just processing emails - you are maintaining professional relationships and ensuring smooth business operations through intelligent, contextual communication.
    """)
    
    if not state["messages"]:
        user_input = "Enhanced email assistant ready! Features: Email monitoring, Calendar integration, Booking automation, Follow-up management. Type 'start monitoring' to begin."
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nAssistant command (start monitoring/stop monitoring/read emails/check calendar/send follow-ups/quit): ")
        print(f"\n👤 USER: {user_input}")
        
        if user_input.strip().lower() == "quit":
            return {"messages": list(state["messages"]) + [HumanMessage(content="quit")]}
        
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 ASSISTANT: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 EXECUTING: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Check if user wants to quit
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and message.content.lower().strip() == "quit":
            return "end"
        
    return "continue"

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

app = graph.compile()

def run_email_assistant():
    print("\n" + "="*60)
    print(" ENHANCED AUTONOMOUS EMAIL ASSISTANT")
    print("="*60)
    print("🎯 Features: Email Monitoring + Calendar Integration + Booking Automation")
    print("📧 Personalized Responses + Follow-up Management")
    print("⚡ Optimized for professional communication")
    
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    status_messages = []
    if vectorstore:
        status_messages.append("✅ Company database connected")
    else:
        status_messages.append("⚠️ Company database not available - using enhanced templates")
    
    if calendar_service:
        status_messages.append("✅ Google Calendar integration active")
    else:
        status_messages.append("⚠️ Calendar integration not available - email-only mode")
    
    for msg in status_messages:
        print(msg)
    
    print("\n🚀 Starting Enhanced Email Assistant...")
    
    try:
        state = {"messages": []}
        
        for step in app.stream(state, stream_mode="values"):
            if "messages" in step:
                messages = step["messages"]
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, ToolMessage):
                        print(f"\n🛠️ RESULT: {last_message.content[:200]}{'...' if len(last_message.content) > 200 else ''}")
                
    except KeyboardInterrupt:
        print("\n\n👋 Enhanced email assistant stopped by user.")
    except Exception as e:
        print(f"\n❌ Error running email assistant: {e}")
    
    print("\n" + "="*60)
    print(" EMAIL ASSISTANT SESSION ENDED")
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
        run_email_assistant()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()