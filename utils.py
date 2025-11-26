# utils.py
from datetime import datetime, date as ddate, time as dtime, timedelta
import os
import joblib

# =========================================================
#  BASIC HELPERS
# =========================================================

def combine_date_time(d: ddate, t: dtime):
    if d is None and t is None:
        return None
    if d is None:
        d = datetime.utcnow().date()
    if t is None:
        t = dtime.min
    return datetime.combine(d, t)

def save_uploaded_file(uploaded_file, lead_id, folder_name="uploaded_invoices"):
    if uploaded_file is None:
        return None
    folder = os.path.join(os.getcwd(), folder_name)
    os.makedirs(folder, exist_ok=True)
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    full_path = os.path.join(folder, fname)
    with open(full_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return full_path

def format_currency(val, currency="$"):
    try:
        if val is None:
            return f"{currency}0"
        return f"{currency}{float(val):,.2f}"
    except:
        return f"{currency}{val}"

# =========================================================
#  SLA / PRIORITY LOGIC
# =========================================================

def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or 24))
        remaining = deadline - datetime.utcnow()
        return remaining.total_seconds(), remaining.total_seconds() <= 0
    except:
        return float("inf"), False


def remaining_sla_hms(seconds):
    if seconds is None or seconds == float("inf"):
        return "—"
    if seconds <= 0:
        return "00:00:00"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def compute_priority_for_lead_row(lead_row, weights):
    val = float(lead_row.get("estimated_value") or 0.0)
    baseline = weights.get("value_baseline", 5000.0)
    value_score = min(val / baseline, 1.0)

    try:
        sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
        if sla_entered is None:
            time_left_h = 9999.0
        else:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
            time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except:
        time_left_h = 9999.0

    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)

    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0

    urgency_component = (
        contacted_flag * weights.get("contacted_w", 0.6) +
        inspection_flag * weights.get("inspection_w", 0.5) +
        estimate_flag * weights.get("estimate_w", 0.5)
    )

    total_weight = (
        weights.get("value_weight", 0.5) +
        weights.get("sla_weight", 0.35) +
        weights.get("urgency_weight", 0.15)
    )

    if total_weight <= 0:
        total_weight = 1.0

    score = (
        value_score * weights.get("value_weight", 0.5) +
        sla_score * weights.get("sla_weight", 0.35) +
        urgency_component * weights.get("urgency_weight", 0.15)
    ) / total_weight

    score = max(0.0, min(score, 1.0))
    return score


# =========================================================
#  ML FUNCTION (placeholder OR real model if available)
# =========================================================

MODEL_PATH = "lead_conversion_model.pkl"

def predict_lead_pr(lead_row):
    """
    Returns the predicted probability of conversion.
    If the ML model does not exist, returns None gracefully.
    """
    if not os.path.exists(MODEL_PATH):
        return None  # allow Streamlit to run without ML

    try:
        model = joblib.load(MODEL_PATH)
    except:
        return None

    try:
        features = [
            float(lead_row.get("estimated_value") or 0),
            float(lead_row.get("contacted") or 0),
            float(lead_row.get("inspection_scheduled") or 0),
            float(lead_row.get("estimate_submitted") or 0),
        ]
        proba = model.predict_proba([features])[0][1]
        return round(float(proba), 3)
    except:
        return None

