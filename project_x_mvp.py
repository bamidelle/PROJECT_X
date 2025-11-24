# project_x_all_in_one.py
"""
Project X — ALL-IN-ONE Streamlit app (Option B: Everything included)

- Single-file Streamlit app with SQLite + SQLAlchemy
- Full feature set merged (Leads capture, Pipeline Board, Analytics, Exports)
- Pipeline Board upgraded with Google Ads–style dashboard:
    * Clickable stage cards (filters)
    * CSS animations & hover effects
    * Auto-refresh (configurable, safe)
    * SLA countdown badges
    * Lightweight "Kanban-like" move-left/move-right controls (no external libs)
- Safe ML integration using `predict_lead_probability(model, row)` if `lead_conversion_model.pkl` exists
Run:
    streamlit run project_x_all_in_one.py
"""

import os
from datetime import datetime, timedelta, date as _date, time as _time
import io
import math

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

MODEL_PATH = "lead_conversion_model.pkl"

# ---------------------------
# APP CSS
# ---------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#0b0f13;
  --muted:#93a0ad;
  --white:#ffffff;
  --money-green:#22c55e;
  --primary-blue:#2563eb;
  --accent-orange:#f97316;
  --danger:#ef4444;
}
body, .stApp {
  background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%);
  color: var(--white);
  font-family: 'Roboto', sans-serif;
}
.header { padding: 12px; color: var(--white); font-weight:600; font-size:18px; }
.metric-card { padding: 18px; border-radius: 12px; color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.22); text-align: left; margin-bottom: 12px; transition: transform 0.12s ease;}
.metric-card:hover { transform: translateY(-4px); }
.metric-label { font-size: 13px; color: #93a0ad; text-transform:uppercase; letter-spacing:0.6px;}
.metric-value { font-size: 32px; font-weight:700; margin-top:6px; }
.progress-bar { width:100%; height:8px; background: rgba(255,255,255,0.06); border-radius:4px; overflow:hidden; margin-top:10px; }
.progress-fill { height:100%; border-radius:4px; transition: width 0.3s ease; }
.stage-badge { padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-left:8px; }
.priority-card { margin:10px 0; padding:18px; border-radius:12px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); }
.small-muted { color:#93a0ad; font-size:13px; }
.btn-mini { padding:6px 10px; border-radius:8px; border:none; cursor:pointer; font-weight:600; }
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
def combine_date_time(d: _date, t: _time):
    if d is None and t is None:
        return None
    if d is None:
        d = datetime.utcnow().date()
    if t is None:
        t = _time.min
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
        if val is None or (isinstance(val, float) and pd.isna(val)):
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
def predict_lead_probability(model, row):
    try:
        if hasattr(row, "to_dict"):
            rd = row.to_dict()
        elif isinstance(row, dict):
            rd = row
        else:
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

# ========== REPLACE your existing Pipeline Board block with the code below ==========
# Replace starting at:
#    elif page == "Pipeline Board":
# and end before:
#    elif page == "Analytics & SLA":
#
# (The block below is self-contained and includes interactive features chosen)

elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard (Google Ads style — interactive)")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
        st.stop()

    df = leads_df(s)
    weights = st.session_state.weights

    # Load model safely if present
    try:
        lead_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    except Exception:
        lead_model = None

    # -------------------------
    # session-state: filters & autorefresh
    # -------------------------
    if "pipeline_stage_filter" not in st.session_state:
        st.session_state.pipeline_stage_filter = None
    if "pipeline_autorefresh" not in st.session_state:
        st.session_state.pipeline_autorefresh = True
    if "pipeline_last_refresh" not in st.session_state:
        st.session_state.pipeline_last_refresh = datetime.utcnow()

    AUTORELOAD_INTERVAL_SEC = 30

    # top control row (filter + toggle)
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 1, 1, 1])
    with ctrl1:
        st.markdown("**Filter Pipeline** — click a stage card below to filter leads by stage. Click again to clear.")
    with ctrl2:
        if st.button("Clear filter"):
            st.session_state.pipeline_stage_filter = None
    with ctrl3:
        ar = st.checkbox("Auto-refresh (30s)", value=st.session_state.pipeline_autorefresh)
        st.session_state.pipeline_autorefresh = bool(ar)
    with ctrl4:
        st.markdown(f"<div class='small-muted'>Last refresh: {st.session_state.pipeline_last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)

    # auto-refresh logic (safe): rerun if enabled and interval passed
    if st.session_state.pipeline_autorefresh:
        now = datetime.utcnow()
        elapsed = (now - st.session_state.pipeline_last_refresh).total_seconds()
        if elapsed >= AUTORELOAD_INTERVAL_SEC:
            st.session_state.pipeline_last_refresh = now
            st.experimental_rerun()

    # ---------- GOOGLE ADS UI STYLE ----------
    st.markdown("""
    <style>
    .metric-card { background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius:12px; padding:14px; }
    .metric-card:hover { transform: translateY(-4px); }
    .stage-click { cursor: pointer; }
    </style>
    """, unsafe_allow_html=True)

    # ---------- KPI METRICS ----------
    total_leads = len(df)
    qualified_leads = len(df[df["qualified"] == True]) if "qualified" in df else 0
    total_value = df["estimated_value"].sum() if "estimated_value" in df and not df["estimated_value"].isna().all() else 0.0
    awarded_count = len(df[df["status"] == LeadStatus.AWARDED])
    lost_count = len(df[df["status"] == LeadStatus.LOST])
    closed_total = awarded_count + lost_count
    conversion_rate = (awarded_count / closed_total * 100) if closed_total else 0.0
    active_leads = total_leads - closed_total

    st.markdown("### 📊 Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Total Leads</div><div class='metric-value' style='color:#2563eb;'>{total_leads}</div><div class='small-muted'>↑ {qualified_leads} Qualified</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Pipeline Value</div><div class='metric-value' style='color:#22c55e;'>{format_currency(total_value)}</div><div class='small-muted'>Active</div></div>""", unsafe_allow_html=True)
    with k3:
        status_cls = "positive" if conversion_rate > 50 else "negative"
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Conversion Rate</div><div class='metric-value' style='color:#a855f7;'>{conversion_rate:.1f}%</div><div class='metric-change {status_cls}'>{awarded_count}/{closed_total} Won</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class='metric-card'><div class='metric-label'>Active Leads</div><div class='metric-value' style='color:#f97316;'>{active_leads}</div><div class='small-muted'>In Progress</div></div>""", unsafe_allow_html=True)

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

    # clickable stage cards (one column per stage)
    cols = st.columns(len(LeadStatus.ALL))
    for i, stage in enumerate(LeadStatus.ALL):
        cnt = stage_counts.get(stage, 0)
        pct = (cnt / total_leads * 100) if total_leads else 0
        color = stage_colors.get(stage, "#fff")
        with cols[i]:
            btn_key = f"stage_btn_{i}"
            # create a tiny html button area — but use st.button for interaction
            if st.button(f"{stage} ({cnt})", key=btn_key):
                # toggle filter
                if st.session_state.pipeline_stage_filter == stage:
                    st.session_state.pipeline_stage_filter = None
                else:
                    st.session_state.pipeline_stage_filter = stage
                st.experimental_rerun()
            st.markdown(f"""
                <div style='margin-top:6px;'>
                    <div class='progress-bar'><div class='progress-fill' style='background:{color}; width:{pct}%;'></div></div>
                    <div style='text-align:center; color:#93a0ad; margin-top:6px;'>{pct:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- PRIORITY LEADS (with filtering) ----------
    st.markdown("### 🎯 Priority Leads (Top 8)")

    # Build priority list
    priority_items = []
    for _, row in df.iterrows():
        score, _, _, _, _, _, _ = compute_priority_for_lead_row(row, weights)

        # SLA calculation (robust)
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if isinstance(sla_entered, str):
            try:
                sla_entered = datetime.fromisoformat(sla_entered)
            except:
                sla_entered = datetime.utcnow()
        else:
            if pd.isna(sla_entered):
                sla_entered = datetime.utcnow()

        sla_hours = int(row.get("sla_hours") or 24)
        deadline = sla_entered + timedelta(hours=sla_hours)
        remaining = (deadline - datetime.utcnow()).total_seconds()
        overdue = remaining <= 0
        hours_left = max(int(remaining // 3600), 0)

        prob = None
        if lead_model is not None:
            try:
                prob = predict_lead_probability(lead_model, row)
            except:
                prob = None

        priority_items.append({
            "id": int(row["id"]),
            "name": row.get("contact_name") or "No name",
            "value": float(row.get("estimated_value") or 0.0),
            "score": score,
            "status": row.get("status"),
            "overdue": overdue,
            "time_left": hours_left,
            "prob": prob,
            "damage": row.get("damage_type") or "Unknown"
        })

    pr_df = pd.DataFrame(priority_items).sort_values("score", ascending=False)

    # apply filter if selected
    if st.session_state.pipeline_stage_filter:
        pr_df = pr_df[pr_df["status"] == st.session_state.pipeline_stage_filter]

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        for _, r in pr_df.head(8).iterrows():
            color = "#ef4444" if r["score"] >= 0.7 else ("#f97316" if r["score"] >= 0.45 else "#22c55e")
            label = "🔴 CRITICAL" if r["score"] >= 0.7 else ("🟠 HIGH" if r["score"] >= 0.45 else "🟢 NORMAL")
            status_color = stage_colors.get(r["status"], "#fff")
            sla_html = "<span style='color:#ef4444;'>❗ OVERDUE</span>" if r["overdue"] else f"<span style='color:#2563eb;'>⏳ {r['time_left']}h left</span>"
            conv_html = ""
            if r["prob"] is not None:
                pct = int(r["prob"] * 100)
                conv_color = "#22c55e" if pct > 70 else ("#f97316" if pct > 40 else "#ef4444")
                conv_html = f"<span style='margin-left:10px; color:{conv_color};'>📊 {pct}% Win</span>"

            st.markdown(f"""
            <div class='priority-card'>
                <div style='display:flex; justify-content:space-between;'>
                    <div>
                        <div><b style='color:{color};'>{label}</b> <span class='stage-badge' style='background:{status_color}20; color:{status_color}; border:1px solid {status_color}40;'>{r['status']}</span></div>
                        <div style='font-size:18px; font-weight:700; margin-top:6px;'>#{r['id']} — {r['name']}</div>
                        <div class='small-muted'>{r['damage'].title()} | Est: <span style='color:#22c55e; font-weight:700;'>{format_currency(r['value'])}</span></div>
                        <div style='margin-top:6px; font-size:13px;'>{sla_html} {conv_html}</div>
                    </div>
                    <div style='text-align:right;'>
                        <div style='font-size:36px; font-weight:700; color:{color};'>{r['score']:.2f}</div>
                        <div style='font-size:11px; color:#93a0ad;'>Priority</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- EXPANDABLE LEAD CARDS with lightweight Kanban actions ----------
    st.markdown("### 📋 All Leads")
    # optionally filter main list by stage
    lead_rows = df.copy()
    if st.session_state.pipeline_stage_filter:
        lead_rows = lead_rows[lead_rows["status"] == st.session_state.pipeline_stage_filter]

    for _, lead_row in lead_rows.sort_values("created_at", ascending=False).iterrows():
        lead_id = int(lead_row["id"])
        lead_obj = s.query(Lead).filter(Lead.id == lead_id).first()
        status_color = stage_colors.get(lead_row["status"], "#fff")
        est_val = lead_row.get("estimated_value") or 0.0
        title = f"#{lead_id} — {lead_row.get('contact_name') or 'No name'} — {lead_row.get('damage_type') or 'Unknown'} — {format_currency(est_val)}"
        with st.expander(title, expanded=False):
            colA, colB = st.columns([3, 1])
            with colA:
                st.markdown(f"**Source:** {lead_row.get('source') or '—'}  ·  **Assigned:** {lead_row.get('assigned_to') or '—'}")
                st.markdown(f"**Address:** {lead_row.get('property_address') or '—'}")
                st.markdown(f"**Notes:** {lead_row.get('notes') or '—'}")
                st.markdown(f"**Created:** {pd.to_datetime(lead_row.get('created_at'))}")
            with colB:
                entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
                try:
                    if isinstance(entered, str):
                        entered = datetime.fromisoformat(entered)
                    elif pd.isna(entered):
                        entered = datetime.utcnow()
                except:
                    entered = datetime.utcnow()
                deadline = entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
                remaining = (deadline - datetime.utcnow()).total_seconds()
                if remaining <= 0:
                    st.markdown(f"<div style='color:{'#ef4444'}; font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                else:
                    hrs = int(remaining // 3600)
                    mins = int((remaining % 3600) // 60)
                    st.markdown(f"<div style='color:#2563eb;'>⏳ {hrs}h {mins}m</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='stage-badge' style='background:{status_color}20; color:{status_color}; border:1px solid {status_color}40;'>{lead_row.get('status')}</div>", unsafe_allow_html=True)

            # Quick actions: move left / move right through LeadStatus.ALL
            move_cols = st.columns([1,1,3,3])
            with move_cols[0]:
                if st.button("◀ Move Left", key=f"move_left_{lead_id}"):
                    try:
                        cur_idx = LeadStatus.ALL.index(lead_obj.status)
                        new_idx = max(0, cur_idx - 1)
                        lead_obj.status = LeadStatus.ALL[new_idx]
                        s.add(lead_obj); s.commit()
                        st.experimental_rerun()
                    except Exception:
                        pass
            with move_cols[1]:
                if st.button("Move Right ▶", key=f"move_right_{lead_id}"):
                    try:
                        cur_idx = LeadStatus.ALL.index(lead_obj.status)
                        new_idx = min(len(LeadStatus.ALL)-1, cur_idx + 1)
                        lead_obj.status = LeadStatus.ALL[new_idx]
                        s.add(lead_obj); s.commit()
                        st.experimental_rerun()
                    except Exception:
                        pass
            with move_cols[2]:
                if st.button("📞 Call", key=f"call_{lead_id}"):
                    # link cannot be directly opened from backend; show phone
                    st.warning(f"Call {lead_row.get('contact_phone') or 'No phone on record'}")
            with move_cols[3]:
                if st.button("✉️ Email", key=f"email_{lead_id}"):
                    st.info(f"Send email to: {lead_row.get('contact_email') or 'No email on record'}")

            st.markdown("---")
            # Update form
            with st.form(f"update_lead_form_{lead_id}"):
                new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead_obj.status) if lead_obj.status in LeadStatus.ALL else 0, key=f"status_sel_{lead_id}")
                new_assigned = st.text_input("Assigned to", value=lead_obj.assigned_to or "", key=f"assign_{lead_id}")
                new_est = st.number_input("Estimate amount ($)", min_value=0.0, value=lead_obj.estimated_value or 0.0, step=50.0, key=f"est_{lead_id}")
                contacted = st.checkbox("Contacted", value=bool(lead_obj.contacted), key=f"contacted_{lead_id}")
                inspection_sched = st.checkbox("Inspection scheduled", value=bool(lead_obj.inspection_scheduled), key=f"inspsch_{lead_id}")
                notes_txt = st.text_area("Notes", value=lead_obj.notes or "", key=f"notes_{lead_id}")
                if st.form_submit_button("Save changes", key=f"save_{lead_id}"):
                    lead_db = s.query(Lead).filter(Lead.id == lead_id).first()
                    if lead_db:
                        lead_db.status = new_status
                        lead_db.assigned_to = new_assigned
                        lead_db.estimated_value = float(new_est) if new_est else None
                        lead_db.contacted = bool(contacted)
                        lead_db.inspection_scheduled = bool(inspection_sched)
                        lead_db.notes = notes_txt
                        s.add(lead_db); s.commit()
                    st.success("Lead updated")
                    st.experimental_rerun()

# ========== End of Pipeline Board replacement ==========
# Continue with Analytics & SLA and Exports pages below

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
        colors = ["#2563eb", "#f9ab00", "#f97316", "#14b8a6", "#a855f7", "#22c55e", "#ef4444"]
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
