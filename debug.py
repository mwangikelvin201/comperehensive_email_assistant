import os
from datetime import datetime
from dotenv import load_dotenv
import pytz

# Load .env file
load_dotenv()

# Get system time
now = datetime.now()

# Get timezone from .env or default to UTC
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")
tz_obj = pytz.timezone(DEFAULT_TIMEZONE)

# Localize current time
localized_now = tz_obj.localize(now.replace(tzinfo=None))
utc_now = datetime.utcnow().replace(tzinfo=pytz.UTC)

print("=" * 40)
print("🕒 TIME DEBUG TOOL")
print("=" * 40)
print(f"System time (naive):        {now}")
print(f"UTC now:                    {utc_now}")
print(f"DEFAULT_TIMEZONE (env):     {DEFAULT_TIMEZONE}")
print(f"Localized time:             {localized_now}")
print(f"Localized ISO Format:       {localized_now.isoformat()}")
print("=" * 40)
