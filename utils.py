# utils.py
import os
from datetime import datetime, date, time as dtime

def combine_date_time(d: date, t: dtime):
    """
    Combine a date and time into a datetime object.
    If one is None, use defaults (today / 00:00).
    """
    if d is None and t is None:
        return None
    if d is None:
        d = datetime.utcnow().date()
    if t is None:
        t = dtime.min
    return datetime.combine(d, t)

def save_uploaded_file(uploaded_file, lead_id):
    """
    Save an uploaded file to the 'uploaded_invoices' folder, naming by lead ID + timestamp.
    Returns full path to saved file.
    """
    if uploaded_file is None:
        return None
    folder = os.path.join(os.getcwd(), "uploaded_invoices")
    os.makedirs(folder, exist_ok=True)
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def format_currency(value):
    """
    Format a numeric value as USD currency string.
    """
    if value is None:
        value = 0.0
    return f"${value:,.2f}"

def calculate_remaining_sla(sla_entered_at, sla_hours):
    """
    Compute remaining SLA time as timedelta. Returns None if inputs invalid.
    """
    from datetime import timedelta, datetime
    if sla_entered_at is None:
        sla_entered_at = datetime.utcnow()
    if isinstance(sla_entered_at, str):
        try:
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        except Exception:
            sla_entered_at = datetime.utcnow()
    sla_hours = int(sla_hours or 24)
    deadline = sla_entered_at + timedelta(hours=sla_hours)
    remaining = deadline - datetime.utcnow()
    return remaining, deadline

def remaining_sla_hms(remaining):
    """
    Convert timedelta to hours, minutes, seconds tuple.
    """
    total_sec = int(remaining.total_seconds())
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return hours, minutes, seconds
