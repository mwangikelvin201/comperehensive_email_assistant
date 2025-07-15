from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.system import SystemMessage
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
from datetime import datetime, timedelta, timezone
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
import pytz

load_dotenv()

# Global variables
vectorstore = None
email_monitor_running = False
processed_emails = set()
calendar_service = None

# Default timezone - should be configurable
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

# Google Calendar API scopes
SCOPES = ['https://www.googleapis.com/auth/calendar', 
          'https://www.googleapis.com/auth/contacts.readonly']

# Simplified importance keywords
IMPORTANCE_KEYWORDS = {
    'urgent': 10, 'emergency': 10, 'critical': 10, 'asap': 10,
    'meeting': 9, 'appointment': 9, 'schedule': 8, 'calendar': 8,'question': 7,
    'query': 7, 'request': 7, 'follow-up': 6, 'update': 6,
    'information': 5, 'details': 5,
    'policy': 9, 'hr': 8, 'company': 8, 'procedure': 7, 'handbook': 7,
    'guidelines': 6, 'benefits': 6, 'vacation': 6, 'leave': 6, 'sick': 6,
    'payroll': 6, 'compensation': 6, 'training': 5, 'development':8,
    'benefits': 6, 'training': 5, 'question': 3 , 'rules': 9
}

# Updated response templates with timezone-aware formatting
RESPONSE_TEMPLATES = {
    'company_query': """Dear {name},

Thank you for your inquiry.

{db_content}

If you have any further questions, please don't hesitate to reach out.

Best regards,
HR Team""",
    
    'meeting_confirmation': """Dear {name},

Thank you for your meeting request. I'm pleased to confirm:

📅 Date: {date}
🕐 Time: {time} ({timezone})
⏱️ Duration: {duration}

I've sent you a calendar invitation. Please let me know if you need to reschedule.

Best regards,
HR Team""",
    
    'meeting_conflict': """Dear {name},

Thank you for your meeting request. Unfortunately, the requested time slot is not available. 

I have the following alternative times available:

{available_times}

Please let me know which time works best for you.

Best regards,
HR Team""",
    
    'meeting_parse_error': """Dear {name},

Thank you for your meeting request. I wasn't able to parse the specific date and time from your email. 

Could you please provide the meeting details in one of these formats:
- "Meeting on January 15th at 2:00 PM"
- "Schedule appointment for 01/15/2024 at 14:00"
- "Book meeting for tomorrow at 9 AM"

I'll be happy to schedule it once I have the specific details.

Best regards,
HR Team"""
}

# Calendar keywords
CALENDAR_KEYWORDS = ['meeting', 'appointment', 'schedule', 'calendar', 'book', 'available', 'time']

def get_timezone_obj(timezone_str):
    """Get timezone object from string."""
    try:
        return pytz.timezone(timezone_str)
    except:
        return pytz.UTC

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
        combined_content = "\n\n".join(relevant_content[:2])  # Take top 2 results
        
        # Limit length
        if len(combined_content) > 1000:
            combined_content = combined_content[:1000] + "..."
        
        return combined_content
        
    except Exception as e:
        print(f"❌ Database search error: {e}")
        return "Error accessing company database"

def extract_datetime_from_email(subject, body):
    """Extract date and time from email content using regex or fallback to GPT if needed."""
    text = f"{subject} {body}".lower()

    # Common regex patterns for date and time
    patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:at\s+)?(\d{1,2}:\d{2}(?:\s*[ap]m)?)',
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:at\s+)?(\d{1,2}\s*[ap]m)',
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)(?:\s+\d{2,4})?\s+(?:at\s+)?(\d{1,2}(:\d{2})?\s*[ap]m)',
        r'((?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\s+(?:at\s+)?(\d{1,2}(:\d{2})?\s*[ap]m)',
        r'(tomorrow|today|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\s+(?:at\s+)?(\d{1,2}(:\d{2})?\s*[ap]m)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                if isinstance(matches[0], tuple) and len(matches[0]) >= 2:
                    date_str, time_str = matches[0][:2]
                    return parse_datetime_components(date_str, time_str)
            except Exception as e:
                print(f"❌ Regex match failed: {e}")
                continue

    # 🔁 Fallback: Use LLM to parse natural date/time from free-form text
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_openai import ChatOpenAI
        from dateutil import parser as dt_parser

        print("🧠 Using GPT fallback for datetime parsing...")

        fallback_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
        llm_response = fallback_model.invoke([
            SystemMessage(content="Extract a specific meeting date and time from this email. Respond in a format parsable by Python datetime (e.g. 'July 18, 2025 10:00 AM')."),
            HumanMessage(content=text)
        ])

        guessed_datetime = dt_parser.parse(llm_response.content)
        tz_obj = get_timezone_obj(DEFAULT_TIMEZONE)
        return guessed_datetime.astimezone(tz_obj)

    except Exception as e:
        print(f"❌ GPT fallback parsing failed: {e}")
        return None


def parse_datetime_components(date_str, time_str):
    """Parse date and time components into datetime object."""
    try:
        # Get timezone
        tz_obj = get_timezone_obj(DEFAULT_TIMEZONE)
        
        # Parse date
        if date_str.lower() == "today":
            base_date = datetime.now(tz_obj).date()
        elif date_str.lower() == "tomorrow":
            base_date = (datetime.now(tz_obj) + timedelta(days=1)).date()
        elif "next" in date_str.lower():
            # Handle "next monday", etc.
            weekday = date_str.lower().replace("next ", "")
            base_date = get_next_weekday(weekday, tz_obj)
        elif any(month in date_str.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
            # Natural language date
            base_date = dateutil.parser.parse(date_str).date()
        else:
            # Standard date format
            base_date = dateutil.parser.parse(date_str).date()
        
        # Parse time
        time_str = time_str.lower().replace(" ", "")
        
        if "am" in time_str or "pm" in time_str:
            # 12-hour format
            time_part = dateutil.parser.parse(time_str).time()
        else:
            # 24-hour format or hour only
            if ":" in time_str:
                hour, minute = map(int, time_str.split(":"))
            else:
                hour = int(time_str)
                minute = 0
            time_part = datetime.min.time().replace(hour=hour, minute=minute)
        
        # Combine date and time
        combined_datetime = datetime.combine(base_date, time_part)
        
        # Make timezone aware
        if combined_datetime.tzinfo is None:
            combined_datetime = tz_obj.localize(combined_datetime)
        
        return combined_datetime
        
    except Exception as e:
        print(f"❌ DateTime parsing error: {e}")
        return None

def get_next_weekday(weekday_name, tz_obj):
    """Get the date of the next occurrence of a weekday."""
    weekdays = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    target_weekday = weekdays.get(weekday_name.lower())
    if target_weekday is None:
        return None
    
    today = datetime.now(tz_obj)
    days_ahead = target_weekday - today.weekday()
    
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    
    return (today + timedelta(days=days_ahead)).date()

def is_company_query(subject, body):
    """Check if email is a company policy/procedure query with improved precision."""
    text = f"{subject} {body}".lower()
    
    # Strong company query indicators
    strong_company_keywords = [
        'company policy', 'company policies', 'company procedure', 'company procedures',
        'hr policy', 'hr policies', 'employee handbook', 'company handbook',
        'company guidelines', 'company rules', 'policy regarding', 'procedure for',
        'what is the policy', 'what are the policies', 'clarification on',
        'guidance on', 'information about', 'details about'
    ]
    
    # Check for strong indicators first
    for keyword in strong_company_keywords:
        if keyword in text:
            return True
    
    # Individual company keywords
    company_keywords = [
        'policy', 'policies', 'procedure', 'procedures', 'hr', 'human resources',
        'company', 'benefits', 'handbook', 'guidelines', 'rules', 'regulation',
        'regulations', 'compliance', 'protocol', 'protocols', 'documentation',
        'vacation', 'leave', 'sick leave', 'absence', 'desertion', 'timeoff',
        'payroll', 'compensation', 'salary', 'wage', 'bonus', 'overtime',
        'training', 'development', 'onboarding', 'termination', 'resignation',
        'disciplinary', 'performance', 'evaluation', 'review', 'dress code',
        'workplace', 'safety', 'security', 'confidentiality', 'social media'
    ]
    
    # Question patterns that suggest information seeking
    question_patterns = [
        r'what is\s+(?:the\s+)?(?:company\s+)?(?:policy|procedure|rule|guideline)',
        r'can you (?:please\s+)?(?:clarify|explain|provide|tell)',
        r'i (?:need|would like|want|require)\s+(?:to\s+)?(?:know|understand|clarify)',
        r'seeking\s+(?:clarification|information|guidance)',
        r'help\s+(?:me\s+)?(?:understand|with)',
        r'what\s+(?:are\s+)?(?:the\s+)?(?:requirements|steps|process)',
        r'how\s+(?:do\s+)?(?:i|we|does|should)'
    ]
    
    # Check if we have company keywords
    has_company_keyword = any(keyword in text for keyword in company_keywords)
    
    # Check for question patterns
    has_question_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)
    
    # Information-seeking language
    info_seeking_phrases = [
        'i am writing to', 'i would like to know', 'i need information',
        'please provide', 'can you help', 'i have a question',
        'seeking clarification', 'need guidance', 'require information',
        'would appreciate', 'looking for', 'need to understand'
    ]
    
    has_info_seeking = any(phrase in text for phrase in info_seeking_phrases)
    
    # Return True if we have company context AND (question patterns OR info-seeking language)
    return has_company_keyword and (has_question_pattern or has_info_seeking)


def is_calendar_request(subject, body):
    """Check if email is a calendar/meeting request with improved precision."""
    text = f"{subject} {body}".lower()
    
    # Strong calendar indicators (high confidence)
    strong_calendar_keywords = [
        'schedule a meeting', 'book a meeting', 'set up a meeting',
        'meeting request', 'appointment request', 'calendar invitation',
        'available times', 'time slots', 'book an appointment',
        'schedule an appointment', 'meeting at', 'appointment at'
    ]
    
    # Check for strong indicators first
    for keyword in strong_calendar_keywords:
        if keyword in text:
            return True
    
    # Individual calendar keywords (need more context)
    calendar_keywords = ['meeting', 'appointment', 'schedule', 'calendar', 'book', 'available', 'time']
    
    # Date/time patterns that suggest scheduling
    datetime_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Date formats
        r'\d{1,2}:\d{2}',  # Time formats
        r'\d{1,2}\s*[ap]m',  # AM/PM
        r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',  # Days
        r'(tomorrow|today|next week)',  # Relative dates
        r'(january|february|march|april|may|june|july|august|september|october|november|december)'  # Months
    ]
    
    # Check if we have calendar keywords AND datetime patterns
    has_calendar_keyword = any(keyword in text for keyword in calendar_keywords)
    has_datetime_pattern = any(re.search(pattern, text, re.IGNORECASE) for pattern in datetime_patterns)
    
    # Only return True if we have both calendar intent AND time references
    if has_calendar_keyword and has_datetime_pattern:
        return True
    
    # Additional check: explicit scheduling language
    scheduling_phrases = [
        'when are you available', 'what time works', 'free time',
        'schedule time', 'set up time', 'find time', 'meeting time'
    ]
    
    return any(phrase in text for phrase in scheduling_phrases)



def classify_email_type(subject, body):
    """Classify email type with priority order and better logic."""
    # First check for company queries (higher priority for policy questions)
    if is_company_query(subject, body):
        return "company_query"
    
    # Then check for calendar requests
    elif is_calendar_request(subject, body):
        return "calendar_request"
    
    # Default to general email
    else:
        return "general"

def extract_sender_name(sender_email):
    """Extract clean sender name."""
    name, email_addr = parseaddr(sender_email)
    if name:
        return name.replace('"', '').strip()
    else:
        username = email_addr.split('@')[0]
        return username.replace('.', ' ').replace('_', ' ').title()

def check_time_availability(requested_datetime, duration_hours=1):
    """Check if requested time slot is available."""
    global calendar_service
    
    if not calendar_service:
        return False
    
    try:
        # Convert to UTC for API call
        start_time = requested_datetime.astimezone(pytz.UTC)
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Check for existing events
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=start_time.isoformat(),
            timeMax=end_time.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Check for conflicts
        for event in events:
            event_start = event['start'].get('dateTime', event['start'].get('date'))
            event_end = event['end'].get('dateTime', event['end'].get('date'))
            
            if 'T' in event_start:  # DateTime event
                event_start_dt = dateutil.parser.parse(event_start)
                event_end_dt = dateutil.parser.parse(event_end)
                
                # Check for overlap
                if (start_time < event_end_dt and end_time > event_start_dt):
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Time availability check failed: {e}")
        return False

def get_alternative_times(requested_datetime, duration_hours=1, days_ahead=7):
    """Get alternative time slots near the requested time."""
    global calendar_service
    
    if not calendar_service:
        return []
    
    try:
        alternative_times = []
        base_date = requested_datetime.date()
        requested_time = requested_datetime.time()
        tz_obj = requested_datetime.tzinfo
        
        # Check same day, different times
        for hour_offset in [-1, 1, -2, 2]:
            try:
                new_time = (datetime.combine(base_date, requested_time) + timedelta(hours=hour_offset)).time()
                new_datetime = datetime.combine(base_date, new_time)
                new_datetime = tz_obj.localize(new_datetime) if new_datetime.tzinfo is None else new_datetime
                
                if new_datetime > datetime.now(tz_obj) and check_time_availability(new_datetime, duration_hours):
                    alternative_times.append({
                        'datetime': new_datetime,
                        'formatted': new_datetime.strftime('%A, %B %d at %I:%M %p %Z')
                    })
                    
                    if len(alternative_times) >= 3:
                        break
            except:
                continue
        
        # If not enough alternatives, check next few days at same time
        if len(alternative_times) < 3:
            for day_offset in range(1, days_ahead + 1):
                try:
                    new_date = base_date + timedelta(days=day_offset)
                    new_datetime = datetime.combine(new_date, requested_time)
                    new_datetime = tz_obj.localize(new_datetime) if new_datetime.tzinfo is None else new_datetime
                    
                    if check_time_availability(new_datetime, duration_hours):
                        alternative_times.append({
                            'datetime': new_datetime,
                            'formatted': new_datetime.strftime('%A, %B %d at %I:%M %p %Z')
                        })
                        
                        if len(alternative_times) >= 3:
                            break
                except:
                    continue
        
        return alternative_times
        
    except Exception as e:
        print(f"❌ Alternative times generation failed: {e}")
        return []

def create_calendar_event(title, start_datetime, attendee_email, duration_hours=1):
    """Create a calendar event with proper timezone handling."""
    global calendar_service
    
    if not calendar_service:
        return False
    
    try:
        # Ensure timezone awareness
        if start_datetime.tzinfo is None:
            tz_obj = get_timezone_obj(DEFAULT_TIMEZONE)
            start_datetime = tz_obj.localize(start_datetime)
        
        end_datetime = start_datetime + timedelta(hours=duration_hours)
        
        # Create event
        event = {
            'summary': title,
            'start': {
                'dateTime': start_datetime.isoformat(),
                'timeZone': str(start_datetime.tzinfo),
            },
            'end': {
                'dateTime': end_datetime.isoformat(),
                'timeZone': str(end_datetime.tzinfo),
            },
            'attendees': [{'email': attendee_email}],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},  # 24 hour reminder
                    {'method': 'popup', 'minutes': 10},       # 10 minute reminder
                ],
            },
        }
        
        created_event = calendar_service.events().insert(
            calendarId='primary',
            body=event,
            sendUpdates='all'  # Send invitations to attendees
        ).execute()
        
        print(f"✅ Calendar event created: {created_event.get('id')}")
        return True
        
    except Exception as e:
        print(f"❌ Calendar event creation failed: {e}")
        return False
    
def send_email_html(to_email, subject, html_body):
    """Send HTML-formatted email."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        print("❌ SMTP configuration missing")
        return False

    try:
        msg = MIMEText(html_body, "html")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())

        print(f"✅ HTML Email sent to {to_email}")
        return True

    except Exception as e:
        print(f"❌ HTML Email sending failed: {e}")
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

def extract_meeting_duration(subject, body):
    """Extract meeting duration from email content."""
    text = f"{subject} {body}".lower()
    
    # Look for duration patterns
    duration_patterns = [
        r'(\d+)\s*hours?',
        r'(\d+)\s*h\b',
        r'(\d+)\s*minutes?',
        r'(\d+)\s*mins?\b',
        r'(\d+)\s*m\b'
    ]
    
    for pattern in duration_patterns:
        matches = re.findall(pattern, text)
        if matches:
            duration_value = int(matches[0])
            if 'minute' in pattern or 'min' in pattern or 'm' in pattern:
                return duration_value / 60  # Convert to hours
            else:
                return duration_value
    
    return 1  # Default 1 hour

def process_email_smart(mail, email_id):
    """Process email with improved classification logic."""
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
        print(f"Body preview: {body[:100]}...")  # Debug: show body preview
        
        # Classify email type using improved logic
        email_type = classify_email_type(subject, body)
        
        if email_type == "company_query":
            print("🏢 Detected: Company query")
            handle_company_query(sender_email, sender_name, subject, body)
            
        elif email_type == "calendar_request":
            print("📅 Detected: Calendar/Meeting request")
            handle_calendar_request(sender_email, sender_name, subject, body)
            
        else:
            print("📝 Detected: General email (no auto-response)")
        
    except Exception as e:
        print(f"❌ Email processing error: {e}")

def handle_calendar_request(sender_email, sender_name, subject, body):
    """Handle calendar/meeting requests with proper time parsing."""
    print("🔄 Processing calendar request...")
    
    # Extract requested date and time
    requested_datetime = extract_datetime_from_email(subject, body)
    duration = extract_meeting_duration(subject, body)
    
    if not requested_datetime:
        print("❌ Could not parse date/time from email")
        response_body = RESPONSE_TEMPLATES['meeting_parse_error'].format(name=sender_name)
        send_email_simple(sender_email, f"Re: {subject}", response_body)
        return
    
    print(f"📅 Requested time: {requested_datetime.strftime('%Y-%m-%d %H:%M %Z')}")
    print(f"⏱️ Duration: {duration} hours")
    
    # Check if requested time is available
    if check_time_availability(requested_datetime, duration):
        print("✅ Time slot available")
        
        # Create calendar event
        event_created = create_calendar_event(
            f"Meeting with {sender_name}",
            requested_datetime,
            sender_email,
            duration
        )
        
        if event_created:
            # Send confirmation
            response_body = RESPONSE_TEMPLATES['meeting_confirmation'].format(
                name=sender_name,
                date=requested_datetime.strftime('%A, %B %d, %Y'),
                time=requested_datetime.strftime('%I:%M %p'),
                timezone=str(requested_datetime.tzinfo),
                duration=f"{duration} hour{'s' if duration != 1 else ''}"
            )
            
            send_email_simple(sender_email, f"Meeting Confirmed: {subject}", response_body)
        else:
            print("❌ Failed to create calendar event")
            
    else:
        print("❌ Time slot not available")
        
        # Get alternative times
        alternative_times = get_alternative_times(requested_datetime, duration)
        
        if alternative_times:
            times_text = "\n".join([f"• {slot['formatted']}" for slot in alternative_times])
            response_body = RESPONSE_TEMPLATES['meeting_conflict'].format(
                name=sender_name,
                available_times=times_text
            )
        else:
            response_body = f"""Dear {sender_name},

Thank you for your meeting request. Unfortunately, the requested time slot is not available .

Please suggest a few alternative times that work for you, and I'll be happy to schedule the meeting.

Best regards,
HR Team"""
        
        send_email_simple(sender_email, f"Re: {subject}", response_body)

def handle_company_query(sender_email, sender_name, subject, body):
    """Handle company policy/procedure queries using GPT with HTML formatting."""
    print("🔄 Processing company query...")

    # Clean and build search query
    query = f"{subject} {body}".strip()
    query = re.sub(r'(re:|fwd:|fw:)', '', query, flags=re.IGNORECASE)
    query = re.sub(r'dear\s+\w+', '', query, flags=re.IGNORECASE)
    query = re.sub(r'best\s+regards.*', '', query, flags=re.IGNORECASE)

    # Search Pinecone
    db_chunks = search_company_db(query)

    # Prompt GPT to write a full HTML email
    prompt = f"""
Write a formal, polite company response email in HTML format. It should:
- Start with a greeting ("Dear {sender_name},")
- Use the provided company info to answer the question clearly and helpfully
- End with a courteous sign-off
- Format the email with HTML tags (e.g., <p>, <strong>, <br>), keeping it clean and readable

Employee Name: {sender_name}
Query: {query}

Company Info:
{db_chunks}
"""

    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = model.invoke([
            SystemMessage(content="You are a professional HR assistant who writes clean, formatted HTML emails."),
            HumanMessage(content=prompt)
        ])
        html_body = response.content.strip()

        send_email_html(sender_email, f"Re: {subject}", html_body)

    except Exception as e:
        print(f"❌ GPT HTML generation failed: {e}")
        fallback_text = RESPONSE_TEMPLATES['company_query'].format(name=sender_name, db_content=db_chunks)
        send_email_simple(sender_email, f"Re: {subject}", fallback_text)

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
        print(f"🕐 Timezone: {DEFAULT_TIMEZONE}")
        
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
    
    return f"✅ Email monitoring started!\n🔧 Features: {', '.join(features)}\n⏱️ Check interval: {check_interval}s\n🕐 Timezone: {DEFAULT_TIMEZONE}"

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
    
    return f"📊 Email Monitor Status: {status}\n🔧 Features: {', '.join(features)}\n📈 Processed emails: {len(processed_emails)}\n🕐 Timezone: {DEFAULT_TIMEZONE}"

# Tools list
tools = [start_email_monitoring, stop_email_monitoring, check_email_status]

# Model
model = ChatOpenAI(model="gpt-4o", temperature=0.3).bind_tools(tools)

def email_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""

Instructions:You are an intelligent email assistant that automatically handles answering emails and scheduling meetings with enhanced calendar integration and company database search capabilities:

## 📧 Email Types:
   1. You are an AI assistant that drafts professional emails based on internal company knowledge.
     **Company Queries** - Policy, procedure, HR, rules, benefits, etc.
    Using the context below, write a clear and concise email that addresses the user's intent.
    Ensure proper grammar, a polite tone, and a logical structure.

    CONTEXT:
    {context}

    USER REQUEST:
    {user_query}
   
2. **Calendar Requests** - Meeting, appointment, scheduling requests
   - Parse specific date/time from email content
   - Check calendar availability for exact requested time
   - Book meetings with proper timezone handling,use the specific timezone passed in the code
   - Send confirmations with accurate time information
                                  

## 🎯 Core Features:
- **Auto-categorization** of incoming emails
- **Smart datetime parsing** from natural language
- **Timezone-aware** calendar booking
- **Conflict detection** and alternative time suggestions
- **Comperehensive Database search** for company information
- **Professional response** templates

## 📋 Available Commands:
- `start monitoring` - Begin email monitoring
- `stop monitoring` - Stop email monitoring  
- `check status` - View monitoring status

## 🔧 Enhanced Processing:
- Automatically detects email type and intent
- Extracts specific dates/times from email content
- Respects timezone settings for accurate scheduling
- Provides alternative times when conflicts exist
- Searches company database for policy questions and anser them in full following email etiquette
- Sends professional, personalized responses

## 🕐 Timezone Handling:
- Uses configurable default timezone
- Properly converts times for calendar API
- Displays times in user's local timezone
- Handles daylight saving time transitions

Ready to assist with intelligent email management with enhanced calendar integration!
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
    print("🚀 SMART EMAIL ASSISTANT v2.0")
    print("="*60)
    print("📧 Automatic Email Processing")
    print("📅 Enhanced Calendar Integration")
    print("🏢 Company Database Search")
    print("🕐 Timezone-Aware Scheduling")
    print("="*60)
    
    # Check requirements
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found!")
        return
    
    # Show status
    print(f"📊 Company DB: {'✅ Connected' if pinecone_available else '❌ Not available'}")
    print(f"📅 Calendar: {'✅ Connected' if calendar_available else '❌ Not available'}")
    print(f"🕐 Timezone: {DEFAULT_TIMEZONE}")
    
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