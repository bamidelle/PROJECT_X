# project_x_all_in_one.py
"""
Assan / Project X — ALL-IN-ONE Streamlit app (Option C: Clean Full Feature Version
with predict_lead_probability standardized)

- Single-file Streamlit app with SQLite + SQLAlchemy
- Google Ads–style Pipeline Dashboard integrated
- Migration-safe DB init
- Utilities (SLA, priority scoring, file save, formatting)
- ML placeholder using predict_lead_probability(model, row)
"""

import os
from datetime import datetime, timedelta, date as date_, time as time_
import io

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# CONFIG / DB
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "assan1_app.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# Migration-safe column defs (SQLite-friendly)
MIGRATION_COLUMNS = {
    "contacted": "INTEGER DEFAULT 0",
    "inspection_scheduled": "INTEGER DEFAULT 0",
    "inspection_scheduled_at": "TEXT",
    "inspection_completed": "INTEGER DEFAULT 0",
    "inspection_completed_at": "TEXT",
    "estimate_submitted": "INTEGER DEFAULT 0",
    "estimate_submitted_at": "TEXT",
    "awarded_comment": "TEXT",
    "awarded_date": "TEXT",
    "awarded_invoice": "TEXT",
    "lost_comment": "TEXT",
    "lost_date": "TEXT",
    "qualified": "INTEGER DEFAULT 0"
}

# ---------------------------
# APP CSS
# ---------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#0b0f13;
  --muted:#93a0ad;
  --white:#ffffff;
  --placeholder:#3a3a3a;
  --radius:10px;
  --primary-red:#ff2d2d;
  --money-green:#22c55e;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}

/* Base */
body, .stApp {
  background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%);
  color: var(--white);
  font-family: 'Roboto', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: transparent !important;
  padding: 18px;
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* Header */
.header { padding: 12px; color: var(--white); font-weight:600; font-size:18px; }

/* Money values (green) */
.money { color: var(--money-green); font-weight:700; }

/* Quick contact overrides for inline use */
.quick-call { background: var(--call-blue); color:#000; border-radius:8px; padding:6px 10px; border:none; }
.quick-wa { background: var(--wa-green); color:#000; border-radius:8px; padding:6px 10px; border:none; }

/* KPI metric card */
.metric-card {
    padding: 18px;
    border-radius: 12px;
    color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    text-align: left;
    margin-bottom: 12px;
}
.metric-number { font-size: 34px; font-weight: 700; margin-bottom: -6px; }
.metric-label { font-size: 15px; opacity: 0.95; }

/* Form submit buttons -> red background with black text */
div[data-testid="stFormSubmitButton"] > button, button.stButton[data-testid="stFormSubmitButton"] > button {
  background: var(--primary-red) !important;
  color: #000000 !important;
  border: 1px solid var(--primary-red) !important;
}
"""

# ---------------------------
# MODELS
# ---------------------------
class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    source = Column(String, default="Unknown")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)

    status = Column(String, default=LeadStatus.NEW)
    assigned_to = Column(String, nullable=True)
    estimated_value = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    sla_hours = Column(Integer, default=24)
    sla_stage = Column(String, default=LeadStatus.NEW)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)

    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(Text, nullable=True)
    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)

    estimates = relationship("Estimate", back_populates="lead", cascade="all, delete-orphan")

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    amount = Column(Float, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    approved = Column(Boolean, default=False)
    approved_at = Column(DateTime, nullable=True)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String, nullable=True)
    details = Column(Text, nullable=True)

    lead = relationship("Lead", back_populates="estimates")

# ---------------------------
# DB INIT & MIGRATION
# ---------------------------
def create_tables_and_migrate():
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "leads" not in inspector.get_table_names():
        return
    existing_cols = {c["name"] for c in inspector.get_columns("leads")}
    conn = engine.connect()
    for col, def_sql in MIGRATION_COLUMNS.items():
        if col not in existing_cols:
            try:
                conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {def_sql};")
            except Exception as e:
                print("Migration add column failed:", col, e)
    conn.close()

def init_db():
    create_tables_and_migrate()

# ---------------------------
# DB HELPERS
# ---------------------------
def get_session():
    return SessionLocal()

def add_lead(session, **kwargs):
    lead = Lead(
        source=kwargs.get("source"),
        source_details=kwargs.get("source_details"),
        contact_name=kwargs.get("contact_name"),
        contact_phone=kwargs.get("contact_phone"),
        contact_email=kwargs.get("contact_email"),
        property_address=kwargs.get("property_address"),
        damage_type=kwargs.get("damage_type"),
        status=LeadStatus.NEW,
        assigned_to=kwargs.get("assigned_to"),
        estimated_value=kwargs.get("estimated_value"),
        notes=kwargs.get("notes"),
        sla_hours=kwargs.get("sla_hours", 24),
        sla_stage=LeadStatus.NEW,
        sla_entered_at=kwargs.get("sla_entered_at") or datetime.utcnow(),
        contacted=kwargs.get("contacted", False),
        inspection_scheduled=kwargs.get("inspection_scheduled", False),
        inspection_scheduled_at=kwargs.get("inspection_scheduled_at"),
        inspection_completed=kwargs.get("inspection_completed", False),
        inspection_completed_at=kwargs.get("inspection_completed_at"),
        estimate_submitted=kwargs.get("estimate_submitted", False),
        estimate_submitted_at=kwargs.get("estimate_submitted_at"),
        awarded_comment=kwargs.get("awarded_comment"),
        awarded_date=kwargs.get("awarded_date"),
        awarded_invoice=kwargs.get("awarded_invoice"),
        lost_comment=kwargs.get("lost_comment"),
        lost_date=kwargs.get("lost_date"),
        qualified=kwargs.get("qualified", False),
    )
    session.add(lead)
    session.commit()
    return lead

def leads_df(session):
    import pandas as _pd
    return _pd.read_sql(session.query(Lead).statement, session.bind)

def estimates_df(session):
    import pandas as _pd
    return _pd.read_sql(session.query(Estimate).statement, session.bind)

def create_estimate(session, lead_id, amount, details=""):
    est = Estimate(lead_id=lead_id, amount=amount, details=details, created_at=datetime.utcnow())
    session.add(est)
    session.commit()
    # update lead
    lead = session.query(Lead).filter(Lead.id == lead_id).first()
    if lead:
        lead.estimated_value = float(amount)
        lead.estimate_submitted = True
        lead.estimate_submitted_at = datetime.utcnow()
        lead.status = LeadStatus.ESTIMATE_SUBMITTED
        session.add(lead)
        session.commit()
    return est

def mark_estimate_sent(session, estimate_id):
    est = session.query(Estimate).filter(Estimate.id == estimate_id).first()
    if est:
        est.sent_at = datetime.utcnow()
        session.add(est); session.commit()
    return est

def mark_estimate_approved(session, estimate_id):
    est = session.query(Estimate).filter(Estimate.id == estimate_id).first()
    if est:
        est.approved = True
        est.approved_at = datetime.utcnow()
        session.add(est)
        lead = est.lead
        lead.status = LeadStatus.AWARDED
        lead.awarded_date = datetime.utcnow()
        session.add(lead)
        session.commit()
    return est

def mark_estimate_lost(session, estimate_id, reason="Lost"):
    est = session.query(Estimate).filter(Estimate.id == estimate_id).first()
    if est:
        est.lost = True
        est.lost_reason = reason
        session.add(est)
        lead = est.lead
        lead.status = LeadStatus.LOST
        lead.lost_date = datetime.utcnow()
        session.add(lead)
        session.commit()
    return est

# ---------------------------
# UTILITIES
# ---------------------------
def combine_date_time(d: date_, t: time_):
    if d is None and t is None:
        return None
    if d is None:
        d = datetime.utcnow().date()
    if t is None:
        t = time_.min
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
        if val is None or (isinstance(val, float) and (pd.isna(val))):
            return f"{currency}0.00"
        return f"{currency}{float(val):,.2f}"
    except Exception:
        return f"{currency}{val}"

def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or 24))
        remaining = deadline - datetime.utcnow()
        return remaining.total_seconds(), remaining.total_seconds() <= 0
    except Exception:
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
    """
    Returns: (score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h)
    """
    val = float(lead_row.get("estimated_value") or 0.0)
    baseline = weights.get("value_baseline", 5000.0)
    value_score = min(val / baseline, 1.0)

    try:
        sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
        if sla_entered is None:
            time_left_h = 9999.0
        else:
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
            time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except Exception:
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
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

# ---------------------------
# ML placeholder (standardized name)
# ---------------------------
MODEL_PATH = "lead_conversion_model.pkl"

def predict_lead_probability(model, row):
    """
    Standardized function name requested by user.
    Accepts:
      - model: a scikit-learn-like model with predict_proba
      - row: pandas Series or dict with lead fields (estimated_value, contacted, inspection_scheduled, estimate_submitted)
    Returns probability (0..1) or None on failure.
    """
    try:
        # Accept either a pandas Series or a dict-like
        if hasattr(row, "to_dict"):
            rd = row.to_dict()
        elif isinstance(row, dict):
            rd = row
        else:
            # fallback: try to index
            rd = dict(row)

        features = [
            float(rd.get("estimated_value") or 0),
            1.0 if bool(rd.get("contacted")) else 0.0,
            1.0 if bool(rd.get("inspection_scheduled")) else 0.0,
            1.0 if bool(rd.get("estimate_submitted")) else 0.0,
        ]
        proba = model.predict_proba([features])[0][1]
        return float(proba)
    except Exception:
        return None

def predict_lead_probability_safe(row):
    """Load model if exists and return predicted probability or None."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        return None
    return predict_lead_probability(model, row)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Assan — CRM", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()
st.markdown("<div class='header'>Assan — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar: control + priority tuning
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

    # priority tuning stored in session_state
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }
    st.markdown("### Priority weight tuning")
    st.session_state.weights["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("Within urgency flags:")
    st.session_state.weights["contacted_w"] = st.slider("Not-contacted weight", 0.0, 1.0, float(st.session_state.weights["contacted_w"]), step=0.05)
    st.session_state.weights["inspection_w"] = st.slider("Not-scheduled weight", 0.0, 1.0, float(st.session_state.weights["inspection_w"]), step=0.05)
    st.session_state.weights["estimate_w"] = st.slider("No-estimate weight", 0.0, 1.0, float(st.session_state.weights["estimate_w"]), step=0.05)
    st.session_state.weights["value_baseline"] = st.number_input("Value baseline", min_value=100.0, value=float(st.session_state.weights["value_baseline"]), step=100.0)
    st.markdown('<small class="kv">Tip: Increase SLA weight to prioritise leads nearing deadline; increase value weight to prioritise larger jobs. (Live countdown updates when app reruns / user interacts)</small>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", source_details="gclid=demo",
                 contact_name="Demo Customer", contact_phone="+15550000", contact_email="demo@example.com",
                 property_address="100 Demo Ave", damage_type="water",
                 assigned_to="Alex", estimated_value=None, notes="Demo lead", sla_hours=24, qualified=True)
        st.success("Demo lead added")

    st.markdown(f"DB file: <small>{DB_FILE}</small>", unsafe_allow_html=True)

# --- Page: Leads / Capture
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
            contact_name = st.text_input("Contact name", placeholder="John Doe")
            contact_phone = st.text_input("Contact phone", placeholder="+1-555-0123")
            contact_email = st.text_input("Contact email", placeholder="name@example.com")
        with col2:
            property_address = st.text_input("Property address", placeholder="123 Main St, City, State")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to", placeholder="Estimator name")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context...")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            s = get_session()
            lead = add_lead(
                s,
                source=source,
                source_details=source_details,
                contact_name=contact_name,
                contact_phone=contact_phone,
                contact_email=contact_email,
                property_address=property_address,
                damage_type=damage_type,
                assigned_to=assigned_to,
                notes=notes,
                sla_hours=int(sla_hours),
                qualified=True if qualified_choice == "Yes" else False
            )
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# --- Page: Pipeline Board (UPGRADED)
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard (Google Ads Style + Auto Updates)")

    # Database session
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
        st.stop()

    df = leads_df(s)
    weights = st.session_state.weights

    # Load ML model safely
    try:
        lead_model = joblib.load("lead_conversion_model.pkl") if os.path.exists("lead_conversion_model.pkl") else None
    except:
        lead_model = None

    # --------------------------
    # Session-State for Filters
    # --------------------------
    if "pipeline_stage_filter" not in st.session_state:
        st.session_state.pipeline_stage_filter = None  # Show all by default
    if "pipeline_autorefresh" not in st.session_state:
        st.session_state.pipeline_autorefresh = True
    if "pipeline_autorefresh_ms" not in st.session_state:
        st.session_state.pipeline_autorefresh_ms = 30000  # 30s refresh

    # Auto refresh UI
    if st.session_state.pipeline_autorefresh:
        st.experimental_rerun()

    # --------------------------
    # GOOGLE ADS STYLE CSS
    # --------------------------
    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.02) 100%);
            border-radius: 14px;
            padding: 20px;
            margin: 12px 0;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 10px rgba(0,0,0,0.35);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
        }
        .metric-card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 20px rgba(0,0,0,0.45);
        }
        .stage-card-click {
            cursor: pointer;
            transition: 0.2s;
        }
        .stage-card-click:hover {
            opacity: 0.85;
            transform: translateY(-3px);
        }
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.12);
            border-radius: 6px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.4s ease;
        }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------
    # BASIC METRICS
    # --------------------------
    total_leads = len(df)
    qualified_leads = len(df[df["qualified"] == True])
    awarded_leads = len(df[df["status"] == LeadStatus.AWARDED])
    lost_leads = len(df[df["status"] == LeadStatus.LOST])
    total_value = df["estimated_value"].sum()

    closed = awarded_leads + lost_leads
    conversion_rate = (awarded_leads / closed * 100) if closed > 0 else 0
    active = total_leads - closed

    # --------------------------
    # METRIC CARDS
    # --------------------------
    st.subheader("📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af;font-size:12px;">Total Leads</div>
            <div style="font-size:32px;font-weight:700;color:#2563eb;">{total_leads}</div>
            <div style="color:#22c55e;">↑ {qualified_leads} Qualified</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af;font-size:12px;">Pipeline Value</div>
            <div style="font-size:32px;font-weight:700;color:#22c55e;">${total_value:,.0f}</div>
            <div style="color:#22c55e;">↑ Active</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af;font-size:12px;">Conversion Rate</div>
            <div style="font-size:32px;font-weight:700;color:#a855f7;">{conversion_rate:.1f}%</div>
            <div style="color:{'#22c55e' if conversion_rate>50 else '#ef4444'};">
                {awarded_leads}/{closed} Won
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color:#9ca3af;font-size:12px;">Active Leads</div>
            <div style="font-size:32px;font-weight:700;color:#f97316;">{active}</div>
            <div>In Progress</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --------------------------
    # STAGE CARDS (CLICKABLE)
    # --------------------------
    st.subheader("📈 Pipeline Stages (Click to Filter)")

    stage_counts = df["status"].value_counts().to_dict()
    stage_colors = {
        LeadStatus.NEW: "#2563eb",
        LeadStatus.CONTACTED: "#eab308",
        LeadStatus.INSPECTION_SCHEDULED: "#f97316",
        LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
        LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
        LeadStatus.AWARDED: "#22c55e",
        LeadStatus.LOST: "#ef4444"
    }

    cols = st.columns(len(LeadStatus.ALL))

    for i, stg in enumerate(LeadStatus.ALL):
        count = stage_counts.get(stg, 0)
        pct = (count / total_leads * 100) if total_leads else 0
        color = stage_colors[stg]

        with cols[i]:
            if st.button(stg, key=f"stg_btn_{stg}"):
                st.session_state.pipeline_stage_filter = stg

            st.markdown(f"""
            <div class="metric-card stage-card-click">
                <div style="color:#9ca3af;font-size:12px;">{stg}</div>
                <div style="font-size:24px;font-weight:700;color:{color};">{count}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="background:{color}; width:{pct}%;"></div>
                </div>
                <div style="text-align:center;color:#9ca3af;font-size:12px;margin-top:6px;">
                    {pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Apply filter if selected
    if st.session_state.pipeline_stage_filter:
        df = df[df["status"] == st.session_state.pipeline_stage_filter]

    # --------------------------
    # PRIORITY ENGINE
    # --------------------------
    st.subheader("🎯 Priority Leads")

    priority_list = []

    for _, row in df.iterrows():
        score, *_ = compute_priority_for_lead_row(row, weights)

        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if isinstance(sla_entered, str):
            try: sla_entered = datetime.fromisoformat(sla_entered)
            except: sla_entered = datetime.utcnow()

        deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
        remaining = deadline - datetime.utcnow()
        overdue = remaining.total_seconds() <= 0

        # Model prediction
        prob = None
        if lead_model is not None:
            try:
                prob = predict_lead_probability(lead_model, row)
            except:
                prob = None

        priority_list.append({
            "id": int(row["id"]),
            "contact_name": row.get("contact_name"),
            "status": row.get("status"),
            "estimated_value": float(row.get("estimated_value") or 0),
            "priority_score": score,
            "sla_overdue": overdue,
            "time_left_hours": remaining.total_seconds() / 3600,
            "conversion_prob": prob
        })

    pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

    if pr_df.empty:
        st.info("No leads available.")
    else:
        for _, r in pr_df.head(6).iterrows():
            color = "#ef4444" if r["priority_score"] >= 0.7 else "#f97316" if r["priority_score"] >= 0.45 else "#22c55e"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:18px;font-weight:700;color:{color};">Priority {r['priority_score']:.2f}</div>
                <div style="color:#fff;">#{r["id"]} — {r["contact_name"]}</div>
                <div style="color:#9ca3af;font-size:13px;">Est: ${r["estimated_value"]:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")


    # ---------- PIPELINE STAGES ----------
    st.markdown("### 📈 Pipeline Stages")

    stage_colors = {
        LeadStatus.NEW: "#2563eb",
        LeadStatus.CONTACTED: "#eab308",
        LeadStatus.INSPECTION_SCHEDULED: "#f97316",
        LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
        LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
        LeadStatus.AWARDED: "#22c55e",
        LeadStatus.LOST: "#ef4444"
    }

    stage_counts = df["status"].value_counts().to_dict()

    cols = st.columns(len(LeadStatus.ALL))
    for i, stage in enumerate(LeadStatus.ALL):
        count = stage_counts.get(stage, 0)
        pct = (count / total_leads * 100) if total_leads else 0
        color = stage_colors[stage]

        with cols[i]:
            st.markdown(f"""
            <div class='metric-card stage-card'>
                <div class='metric-label'>{stage}</div>
                <div class='metric-value' style='color:{color};'>{count}</div>
                <div class='progress-bar'>
                    <div class='progress-fill' style='background:{color}; width:{pct}%'></div>
                </div>
                <div style='margin-top:6px; font-size:12px; color:#93a0ad; text-align:center;'>
                    {pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- PRIORITY LEADS ----------
# ==================== PRIORITY LEADS CALCULATION ====================
st.markdown("### 🎯 Priority Leads (Top 8)")

priority_list = []
weights = st.session_state.weights

for _, row in df.iterrows():
    score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
    
    # SLA calculation
    sla_entered = row.get("sla_entered_at") or row.get("created_at")
    if isinstance(sla_entered, str):
        try: 
            sla_entered = datetime.fromisoformat(sla_entered)
        except: 
            sla_entered = datetime.utcnow()
    elif pd.isna(sla_entered):
        sla_entered = datetime.utcnow()
        
    deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
    remaining = deadline - datetime.utcnow()
    overdue = remaining.total_seconds() <= 0
    
    priority_list.append({
        "id": int(row["id"]),
        "contact_name": row.get("contact_name") or "No name",
        "estimated_value": float(row.get("estimated_value") or 0.0),
        "time_left_hours": float(remaining.total_seconds() / 3600.0),
        "priority_score": score,
        "status": row.get("status"),
        "sla_overdue": overdue,
        "sla_deadline": deadline,
        "damage_type": row.get("damage_type", "Unknown")
    })

pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

# ==================== PRIORITY LEADS DISPLAY (FIXED) ====================
# REPLACE your entire existing "Pipeline Board" block with the code below.
# Insert this replacement starting exactly at:
#   elif page == "Pipeline Board":
# and ending just before the next `elif page == "Analytics & SLA":` block.

elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard (Google Ads style — interactive)")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    # If no leads, short-circuit (keeps UI consistent)
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
        st.stop()

    # Dataframe for calculations
    df = leads_df(s)
    weights = st.session_state.weights

    # Try load model (optional; safe)
    try:
        lead_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    except Exception:
        lead_model = None

    # -------------------------
    # Session-state: filters & autorefresh
    # -------------------------
    if "pipeline_stage_filter" not in st.session_state:
        st.session_state.pipeline_stage_filter = None  # None -> show all
    if "pipeline_autorefresh" not in st.session_state:
        st.session_state.pipeline_autorefresh = True
    if "pipeline_autorefresh_ms" not in st.session_state:
        st.session_state.pipeline_autorefresh_ms = 30000  # 30s

    # optional small control row (filter clear + autorefresh toggle)
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
    with ctrl_col1:
        if st.button("Clear stage filter"):
            st.session_state.pipeline_stage_filter = None
            st.experimental_rerun()
    with ctrl_col2:
        if st.checkbox("Auto-refresh (30s)", value=st.session_state.pipeline_autorefresh, key="ui_autorefresh_toggle"):
            st.session_state.pipeline_autorefresh = True
        else:
            st.session_state.pipeline_autorefresh = False
    with ctrl_col3:
        # quick search / filter input
        txt = st.text_input("Quick search (name, address, damage type)", value="", key="pipeline_quick_search")
        st.session_state.pipeline_quick_search = txt.strip()

    # Add safe JS-based page reload if autorefresh enabled
    if st.session_state.pipeline_autorefresh:
        # reloads the page every N milliseconds (non-blocking)
        st.components.v1.html(
            f"""
            <script>
               const AUTO_REFRESH_MS = {int(st.session_state.pipeline_autorefresh_ms)};
               if (!window._pipeline_autorefresh_registered) {{
                  window._pipeline_autorefresh_registered = true;
                  setInterval(()=>{{ location.reload(); }}, AUTO_REFRESH_MS);
               }}
            </script>
            """,
            height=0,
            scrolling=False,
        )

    # ---------- GOOGLE ADS UI CSS + animations ----------
    st.markdown(
        """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 12px;
        padding: 18px;
        margin: 8px 0;
        border: 1px solid rgba(255,255,255,0.04);
        box-shadow: 0 6px 18px rgba(0,0,0,0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        cursor: pointer;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(0,0,0,0.28); }
    .metric-label { font-size:12px; color:#93a0ad; text-transform:uppercase; letter-spacing:0.6px; }
    .metric-value { font-size:28px; font-weight:700; margin-top:6px; }
    .metric-change { font-size:12px; padding:4px 8px; border-radius:6px; display:inline-block; font-weight:600; }
    .positive { background: rgba(34,197,94,0.12); color:#22c55e; }
    .negative { background: rgba(239,68,68,0.12); color:#ef4444; }

    .stage-card { padding:12px; border-radius:10px; transition: transform .15s; }
    .stage-card:hover { transform: translateY(-3px); }

    .progress-bar { width:100%; height:8px; background:rgba(255,255,255,0.06); border-radius:6px; overflow:hidden; margin-top:8px; }
    .progress-fill { height:100%; border-radius:6px; transition: width .25s ease; }

    .priority-card { margin:12px 0; padding:14px; border-radius:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); }
    .stage-badge { padding:6px 12px; border-radius:20px; font-size:12px; font-weight:700; margin-left:10px; display:inline-block; opacity:0.95; }
    .sla-badge { padding:6px 10px; border-radius:12px; font-weight:700; font-size:12px; display:inline-block; }

    /* subtle fade-in for new elements */
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .metric-card, .priority-card { animation: fadeInUp .35s ease both; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- KPI METRICS (interactive: clicking a card filters by common group) ----------
    total_leads = len(df)
    qualified_leads = len(df[df["qualified"] == True]) if "qualified" in df else 0
    total_value = float(df["estimated_value"].sum()) if "estimated_value" in df else 0.0
    awarded_count = len(df[df["status"] == LeadStatus.AWARDED]) if "status" in df else 0
    lost_count = len(df[df["status"] == LeadStatus.LOST]) if "status" in df else 0
    closed_total = awarded_count + lost_count
    conversion_rate = (awarded_count / closed_total * 100) if closed_total else 0
    active_leads = total_leads - closed_total

    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)

    # helper to create clickable div (uses query param to store selection)
    def _set_stage_filter(stage_name):
        st.session_state.pipeline_stage_filter = stage_name
        st.experimental_rerun()

    with k1:
        # Total leads clickable -> clears filter (show all)
        if st.button(f"Total Leads: {total_leads}", key="btn_total_leads"):
            _set_stage_filter(None)
        st.markdown(f"<div class='metric-change positive'>↑ {qualified_leads} Qualified</div>", unsafe_allow_html=True)

    with k2:
        if st.button(f"Pipeline Value: {format_currency(total_value)}", key="btn_pipeline_value"):
            _set_stage_filter(None)
        st.markdown("<div class='metric-change positive'>Active</div>", unsafe_allow_html=True)

    with k3:
        if st.button(f"Conversion: {conversion_rate:.1f}%", key="btn_conversion"):
            _set_stage_filter(None)
        st.markdown(f"<div class='metric-change {'positive' if conversion_rate>50 else 'negative'}'>{awarded_count}/{closed_total} Won</div>", unsafe_allow_html=True)

    with k4:
        if st.button(f"Active Leads: {active_leads}", key="btn_active"):
            _set_stage_filter(None)
        st.markdown("<div class='metric-change positive'>In Progress</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- STAGE CARDS (clicking a stage filters to that stage) ----------
    st.markdown("### 📈 Pipeline Stages (click a card to filter)")
    stage_colors = {
        LeadStatus.NEW: "#2563eb",
        LeadStatus.CONTACTED: "#eab308",
        LeadStatus.INSPECTION_SCHEDULED: "#f97316",
        LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
        LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
        LeadStatus.AWARDED: "#22c55e",
        LeadStatus.LOST: "#ef4444",
    }

    stage_counts = df["status"].value_counts().to_dict() if "status" in df else {}
    stage_cols = st.columns(len(LeadStatus.ALL))
    for idx, stage in enumerate(LeadStatus.ALL):
        count = stage_counts.get(stage, 0)
        pct = (count / total_leads * 100) if total_leads else 0
        color = stage_colors.get(stage, "#ffffff")

        with stage_cols[idx]:
            # make stage card clickable: use button that sets session_state filter
            if st.button(f"{stage} — {count}", key=f"stage_btn_{idx}"):
                st.session_state.pipeline_stage_filter = stage
                st.experimental_rerun()

            # visual progress
            st.markdown(
                f"""
                <div class='metric-card stage-card' style='text-align:center;'>
                    <div class='metric-label'>{stage}</div>
                    <div class='metric-value' style='color:{color};'>{count}</div>
                    <div class='progress-bar'><div class='progress-fill' style='background:{color}; width:{pct}%;'></div></div>
                    <div style='font-size:12px; color:#93a0ad; margin-top:6px;'>{pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---------- Apply filters (stage & quick search) ----------
    df_filtered = df.copy()
    # stage filter from session
    if st.session_state.pipeline_stage_filter:
        df_filtered = df_filtered[df_filtered["status"] == st.session_state.pipeline_stage_filter]
    # quick search across name, address, damage_type
    q = st.session_state.get("pipeline_quick_search", "").lower()
    if q:
        mask = pd.Series(False, index=df_filtered.index)
        for col in ["contact_name", "property_address", "damage_type"]:
            if col in df_filtered:
                mask = mask | df_filtered[col].fillna("").str.lower().str.contains(q)
        df_filtered = df_filtered[mask]

    # ---------- PRIORITY LEADS (top 8) with SLA badges and move-status control ----------
    st.markdown("### 🎯 Priority Leads (Top 8)")

    priority_items = []
    for _, row in df_filtered.iterrows():
        score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)

        # SLA
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        try:
            sla_dt = pd.to_datetime(sla_entered)
        except Exception:
            sla_dt = datetime.utcnow()
        sla_hours = int(row.get("sla_hours") or 24)
        deadline = sla_dt + timedelta(hours=sla_hours)
        remaining_secs = (deadline - datetime.utcnow()).total_seconds()
        overdue = remaining_secs <= 0
        hours_left = max(int(remaining_secs // 3600), 0)
        mins_left = max(int((remaining_secs % 3600) // 60), 0)

        # prediction (safe)
        prob = None
        if lead_model is not None:
            try:
                prob = predict_lead_probability(lead_model, row)
            except Exception:
                prob = None

        priority_items.append({
            "id": int(row["id"]),
            "name": row.get("contact_name") or "No name",
            "value": float(row.get("estimated_value") or 0.0),
            "score": score,
            "status": row.get("status"),
            "overdue": overdue,
            "hours_left": hours_left,
            "mins_left": mins_left,
            "prob": prob,
            "damage": row.get("damage_type") or "Unknown",
            "row": row,
        })

    pr_df = pd.DataFrame(priority_items).sort_values("score", ascending=False)

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        for _, r in pr_df.head(8).iterrows():
            color = "#ef4444" if r["score"] >= 0.7 else ("#f97316" if r["score"] >= 0.45 else "#22c55e")
            label = "🔴 CRITICAL" if r["score"] >= 0.7 else ("🟠 HIGH" if r["score"] >= 0.45 else "🟢 NORMAL")
            status_color = stage_colors.get(r["status"], "#ffffff")

            sla_html = (f"<span class='sla-badge' style='background:#ef444420;color:#ef4444;'>❗ OVERDUE</span>"
                        if r["overdue"] else f"<span class='sla-badge' style='background:#2563eb20;color:#2563eb;'>⏳ {r['hours_left']}h {r['mins_left']}m</span>")

            conv_html = ""
            if r["prob"] is not None:
                pct = int(r["prob"] * 100)
                conv_color = "#22c55e" if pct > 70 else ("#f97316" if pct > 40 else "#ef4444")
                conv_html = f"<span style='margin-left:10px;color:{conv_color};font-weight:700;'>📊 {pct}% Win</span>"

            st.markdown(
                f"""
                <div class='priority-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <div style='flex:1;'>
                            <div style='margin-bottom:8px;'>
                                <span style='color:{color};font-weight:700;font-size:13px;'>{label}</span>
                                <span class='stage-badge' style='background:{status_color}30;color:{status_color};border:1px solid {status_color}40;'>{r['status']}</span>
                            </div>
                            <div style='font-size:16px;font-weight:700;'>#{r['id']} — {r['name']}</div>
                            <div style='color:#93a0ad;font-size:13px;margin-top:6px;'>{r['damage'].title()} | Est: <span style='color:#22c55e;font-weight:700;'>${r['value']:,.0f}</span></div>
                            <div style='margin-top:8px;font-size:13px;'>{sla_html} {conv_html}</div>
                        </div>
                        <div style='text-align:right;padding-left:18px;'>
                            <div style='font-size:32px;font-weight:700;color:{color};'>{r['score']:.2f}</div>
                            <div style='font-size:11px;color:#93a0ad;'>Priority</div>
                        </div>
                    </div>

                    <!-- Inline actions: quick-change status form -->
                    <div style='margin-top:10px; display:flex; gap:8px;'>
                    """
            , unsafe_allow_html=True)

            # Inline status change (safe, no drag-and-drop)
            cols = st.columns([2, 1, 1, 1, 3])
            with cols[0]:
                new_status = st.selectbox(f"Move #{r['id']} to", LeadStatus.ALL, index=LeadStatus.ALL.index(r["status"]) if r["status"] in LeadStatus.ALL else 0, key=f"move_status_{r['id']}")
            with cols[1]:
                if st.button("Move", key=f"move_btn_{r['id']}"):
                    s_db = get_session()
                    lead_db = s_db.query(Lead).filter(Lead.id == r["id"]).first()
                    if lead_db:
                        lead_db.status = new_status
                        if new_status == LeadStatus.INSPECTION_SCHEDULED:
                            lead_db.inspection_scheduled = True
                            lead_db.inspection_scheduled_at = datetime.utcnow()
                        s_db.add(lead_db)
                        s_db.commit()
                        st.success(f"Lead #{r['id']} moved to {new_status}")
                        st.experimental_rerun()
            with cols[2]:
                if st.button("Open", key=f"open_{r['id']}"):
                    # set session filter and scroll to lead in All Leads below (approximate)
                    st.session_state.pipeline_stage_filter = None
                    st.experimental_rerun()
            with cols[3]:
                if st.button("Call", key=f"call_{r['id']}"):
                    # no-op UI action; could open tel: link if desired
                    st.info(f"Dial action for {r['name']} (not implemented in preview)")
            with cols[4]:
                st.write("")  # spacer

            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------- ALL LEADS (expandable) ----------
    st.markdown("### 📋 All Leads (expand to edit)")

    # show summary table first for quick glance (clickable row opens edit)
    display_cols = ["id", "contact_name", "status", "estimated_value", "assigned_to", "created_at"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df_filtered[display_cols].sort_values("created_at", ascending=False).head(200))

    # Expandable detailed cards (editable)
    for lead in s.query(Lead).order_by(Lead.created_at.desc()).all():
        # apply stage filter or quick search if active
        if st.session_state.pipeline_stage_filter and lead.status != st.session_state.pipeline_stage_filter:
            continue
        if st.session_state.pipeline_quick_search:
            qv = st.session_state.pipeline_quick_search.lower()
            if qv not in (lead.contact_name or "").lower() and qv not in (lead.property_address or "").lower() and qv not in (lead.damage_type or "").lower():
                continue

        card_label = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {format_currency(lead.estimated_value or 0)}"
        with st.expander(card_label, expanded=False):
            colA, colB = st.columns([3, 1])
            with colA:
                st.write(f"**Source:** {lead.source or '—'}  •  **Assigned:** {lead.assigned_to or '—'}")
                st.write(f"**Address:** {lead.property_address or '—'}")
                st.write(f"**Notes:** {lead.notes or '—'}")
                st.write(f"**Created:** {lead.created_at}")
            with colB:
                entered = lead.sla_entered_at or lead.created_at
                try:
                    entered_dt = pd.to_datetime(entered)
                except:
                    entered_dt = datetime.utcnow()
                deadline = entered_dt + timedelta(hours=(lead.sla_hours or 24))
                remaining_secs = (deadline - datetime.utcnow()).total_seconds()
                if remaining_secs <= 0:
                    st.markdown("<div style='color:#ef4444;font-weight:700;'>❗ SLA OVERDUE</div>", unsafe_allow_html=True)
                else:
                    h = int(remaining_secs // 3600)
                    m = int((remaining_secs % 3600) // 60)
                    st.markdown(f"<div style='color:#2563eb;font-weight:700;'>⏳ {h}h {m}m left</div>", unsafe_allow_html=True)

            st.markdown("---")
            # quick edit form inside each card
            with st.form(f"edit_lead_form_{lead.id}"):
                st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"card_status_{lead.id}")
                st.text_input("Assigned to", value=lead.assigned_to or "", key=f"card_assign_{lead.id}")
                if st.form_submit_button("Save changes"):
                    s_db = get_session()
                    lead_db = s_db.query(Lead).filter(Lead.id == lead.id).first()
                    if lead_db:
                        lead_db.status = st.session_state.get(f"card_status_{lead.id}", lead.status)
                        lead_db.assigned_to = st.session_state.get(f"card_assign_{lead.id}", lead.assigned_to)
                        s_db.add(lead_db)
                        s_db.commit()
                        st.success(f"Lead #{lead.id} updated")
                        st.experimental_rerun()


# --- Page: Analytics & SLA
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
        funnel.columns = ["stage", "count"]
        st.subheader("Funnel Overview")
        colors = ["#2563eb", "#f9ab00", "#fb8c00", "#00acc1", "#9334e6", "#0f9d58", "#ea4335"]
        fig = px.bar(funnel, x="stage", y="count", title="Leads by Stage", text="count", color="stage", color_discrete_sequence=colors[:len(funnel)])
        fig.update_layout(xaxis_title=None, yaxis_title="Number of Leads", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row["sla_entered_at"]
            try:
                if pd.isna(sla_entered_at) or sla_entered_at is None:
                    sla_entered_at = datetime.utcnow()
                elif isinstance(sla_entered_at, str):
                    sla_entered_at = datetime.fromisoformat(sla_entered_at)
            except Exception:
                sla_entered_at = datetime.utcnow()
            sla_hours = int(row["sla_hours"]) if pd.notna(row["sla_hours"]) else 24
            deadline = sla_entered_at + timedelta(hours=sla_hours)
            remaining = deadline - datetime.utcnow()
            overdue_rows.append({
                "id": row["id"],
                "contact": row["contact_name"],
                "status": row["status"],
                "sla_stage": row["sla_stage"],
                "deadline": deadline,
                "overdue": remaining.total_seconds() <= 0
            })
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.dataframe(df_overdue.sort_values("deadline"))
        else:
            st.info("No SLA overdue leads.")

# --- Page: Exports
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    df_est = estimates_df(s)
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")
