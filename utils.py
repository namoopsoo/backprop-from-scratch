from datetime import datetime
import pytz

def utc_now():
    return datetime.utcnow().replace(tzinfo=pytz.UTC)

def utc_ts(dt):
    return dt.strftime("%Y-%m-%dT%H%M%S")
