# project_x_all_in_one.py
"""
All-in-one Streamlit CRM (clean, fixed, full-feature)
- Single-file app: models, db init, utils, UI
- Editable Lead Capture (create / update inline)
- Pipeline Board: Google Ads–style dashboard (clickable stage cards, CSS animations)
- Auto-refresh (30s), SLA countdown badges, priority scoring
- Optional ML model support (loaded lazily; joblib is optional)
- Safe use of st.experimental_rerun only after DB writes
Run:
    streamlit run project_x_all_in_one.py
"""

import os
from datetime import datetime, timedelta, date as date_, time as time_
import math

import streamlit as st
import pandas as pd
import plotly.express as px

# Lazy ML loader will import joblib only if needed
JOBLIB_AVAILABLE = True
try:
    import joblib  # optional
except Exception:
    JOBLIB_AVAILABLE = False

# SQLAlchemy (local sqlite)
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
  --placeholder:#3a3a3a;
  --radius:10px;
  --primary-red:#ff2d2d;
  --money-green:#22c55e;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}
body, .stApp { background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%); color: var(--white); font-family: 'Roboto', sans-serif; }
section[data-testid="stSidebar"] { background: transparent !important; padding: 18px; border-right: 1px solid rgba(255,255,255,0.03); }
.header { padding: 12px; color: var(--white); font-weight:600; font-size:18px; }
.money { color: var(--money-green); font-weight:700; }
.quick-call { background: var(--call-blue); color:#000; border-radius:8px; padding:6px 10px; border:none; }
.quick-wa { background: var(--wa-green); color:#000; border-radius:8px; padding:6px 10px; border:none; }
.metric-card { padding: 18px; border-radius: 12px; color: white; box-shadow: 0 6px 18px rgba(0,0,0,0.22); text-align: left; margin-bottom: 12px; }
.metric-number { font-size: 34px; font-weight: 700; margin-bottom: -6px; }
.metric-label { font-size: 15px; opacity: 0.95; }
div[data-testid="stFormSubmitButton"] > button, button.stButton[data-testid="stFormSubmitButton"] > button {
  background: var(--primary-red) !important; color: #000000 !important; border: 1px solid var(--primary-red) !important;
}
/* Google-style pipeline additions */
.priority-card { margin:10px 0; padding:18px; border-radius:12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.04); }
.stage-badge { padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-left:10px; }
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
            except Exception:
                pass
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
    return pd.read_sql(session.query(Lead).statement, session.bind)

def estimates_df(session):
    return pd.read_sql(session.query(Estimate).statement, session.bind)

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
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

# ---------------------------
# ML helpers (lazy import)
# ---------------------------
def _load_model_if_exists():
    if not os.path.exists(MODEL_PATH) or not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

def predict_lead_probability(model, row):
    """
    Standardized function name.
    Accepts model (sklearn-like) and row (Series/dict).
    """
    try:
        if model is None:
            return None
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
    model = _load_model_if_exists()
    return predict_lead_probability(model, row)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Assan — CRM", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()
st.markdown("<div class='header'>Assan — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar controls
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
    st.markdown('<small class="kv">Tip: Increase SLA weight to prioritise leads nearing deadline; increase value weight to prioritise larger jobs.</small>', unsafe_allow_html=True)

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
            # safe rerun after DB write
            st.experimental_rerun()

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# --- Page: Pipeline Board (REPLACEMENT START)
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard (Google Ads style — interactive)")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
        st.stop()

    df = leads_df(s)
    weights = st.session_state.weights

    lead_model = _load_model_if_exists()

    # Session-state controls for pipeline
    if "pipeline_stage_filter" not in st.session_state:
        st.session_state.pipeline_stage_filter = None
    if "pipeline_autorefresh" not in st.session_state:
        st.session_state.pipeline_autorefresh = True
    if "pipeline_autorefresh_ms" not in st.session_state:
        st.session_state.pipeline_autorefresh_ms = 30000  # 30 secs

    # Auto-refresh: use st.experimental_memo? We'll use st.experimental_set_query_params + rerun timing
    if st.session_state.pipeline_autorefresh:
        st_autorefresh = st.experimental_get_query_params().get("autorefresh_ts", [None])[0]
        # We don't force a background thread; rely on user interactions and toggle

    # Styles specific to pipeline (animations/shadows)
    st.markdown("""
    <style>
    .metric-card { background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.02) 100%); border-radius:12px; padding:16px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.04); transition: transform .12s ease; }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 10px 30px rgba(0,0,0,0.45); }
    .stage-badge { padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-left:8px; }
    .priority-card { margin:10px 0; padding:16px; border-radius:10px; background: rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.03); }
    </style>
    """, unsafe_allow_html=True)

    # KPI calculations
    total_leads = len(df)
    qualified_leads = len(df[df["qualified"] == True]) if "qualified" in df else 0
    total_value = 0.0
    if "estimated_value" in df:
        try:
            total_value = float(df["estimated_value"].fillna(0).sum())
        except Exception:
            total_value = 0.0
    awarded_count = len(df[df["status"] == LeadStatus.AWARDED]) if "status" in df else 0
    lost_count = len(df[df["status"] == LeadStatus.LOST]) if "status" in df else 0
    closed_total = awarded_count + lost_count
    conversion_rate = (awarded_count / closed_total * 100) if closed_total else 0
    active_leads = total_leads - closed_total

    st.markdown("### 📊 Key Performance Indicators")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div style='font-size:12px;color:#93a0ad;'>Total Leads</div><div style='font-size:28px;font-weight:700;color:#2563eb;'>{total_leads}</div><div style='font-size:12px;color:#22c55e;'>↑ {qualified_leads} Qualified</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div style='font-size:12px;color:#93a0ad;'>Pipeline Value</div><div style='font-size:28px;font-weight:700;color:#22c55e;'>{format_currency(total_value)}</div><div style='font-size:12px;color:#22c55e;'>Active</div></div>", unsafe_allow_html=True)
    with c3:
        posneg = "positive" if conversion_rate > 50 else "negative"
        st.markdown(f"<div class='metric-card'><div style='font-size:12px;color:#93a0ad;'>Conversion Rate</div><div style='font-size:28px;font-weight:700;color:#a855f7;'>{conversion_rate:.1f}%</div><div style='font-size:12px;color:#93a0ad;'>{awarded_count}/{closed_total} Won</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div style='font-size:12px;color:#93a0ad;'>Active Leads</div><div style='font-size:28px;font-weight:700;color:#f97316;'>{active_leads}</div><div style='font-size:12px;color:#93a0ad;'>In Progress</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Stage cards (clickable)
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

    stage_counts = df["status"].value_counts().to_dict() if "status" in df else {}
    cols = st.columns(len(LeadStatus.ALL))
    for i, stage in enumerate(LeadStatus.ALL):
        count = stage_counts.get(stage, 0)
        pct = (count / total_leads * 100) if total_leads else 0
        color = stage_colors.get(stage, "#ffffff")
        with cols[i]:
            btn_html = f"""<div class='metric-card' style='cursor:pointer;'>{stage}<div style='font-size:24px;font-weight:700;color:{color};'>{count}</div><div style='height:8px;background:rgba(255,255,255,0.06);border-radius:4px;margin-top:8px;'><div style='width:{pct}%;background:{color};height:100%;border-radius:4px;'></div></div><div style='text-align:center;color:#93a0ad;margin-top:6px;'>{pct:.1f}%</div></div>"""
            clicked = st.button(stage, key=f"stage_btn_{i}")
            st.markdown(btn_html, unsafe_allow_html=True)
            if clicked:
                # toggle filter
                if st.session_state.pipeline_stage_filter == stage:
                    st.session_state.pipeline_stage_filter = None
                else:
                    st.session_state.pipeline_stage_filter = stage

    st.markdown("---")

    # small controls
    ctrl1, ctrl2 = st.columns([1,3])
    with ctrl1:
        if st.button("Clear filter"):
            st.session_state.pipeline_stage_filter = None
    with ctrl2:
        st.checkbox("Auto-refresh (30s)", value=st.session_state.pipeline_autorefresh, key="pipeline_autorefresh")

    # Implement a simple non-blocking auto-refresh by using query params + rerun on timer
    if st.session_state.pipeline_autorefresh:
        # set a timestamp param so we can "simulate" auto activity
        now_ts = int(datetime.utcnow().timestamp())
        st.experimental_set_query_params(autorefresh_ts=now_ts)
        # note: Streamlit has limited ways to auto refresh; this will not force continuous reload,
        # but every user interaction will update the URL param. For robust auto-refresh, use st_autorefresh from streamlit-extras or st.experimental_rerun in an interval - avoid forcing expensive reruns.

    # Filter dataframe if a stage selected
    if st.session_state.pipeline_stage_filter:
        df_view = df[df["status"] == st.session_state.pipeline_stage_filter].copy()
    else:
        df_view = df.copy()

    # ---------- Priority leads
    st.markdown("### 🎯 Priority Leads (Top 8)")
    priority_items = []
    for _, row in df_view.iterrows():
        score, *_ = compute_priority_for_lead_row(row, weights)
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if isinstance(sla_entered, str):
            try:
                sla_entered = datetime.fromisoformat(sla_entered)
            except:
                sla_entered = datetime.utcnow()
        elif pd.isna(sla_entered):
            sla_entered = datetime.utcnow()
        sla_hours_val = int(row.get("sla_hours") or 24)
        deadline = sla_entered + timedelta(hours=sla_hours_val)
        remaining = deadline - datetime.utcnow()
        overdue = remaining.total_seconds() <= 0
        hours_left = max(int(remaining.total_seconds() // 3600), 0)
        prob = None
        try:
            if lead_model is not None:
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
            "time_left": hours_left,
            "prob": prob,
            "damage": row.get("damage_type") or "Unknown"
        })

    pr_df = pd.DataFrame(priority_items).sort_values("score", ascending=False) if priority_items else pd.DataFrame()

    if pr_df.empty:
        st.info("No priority leads to display.")
    else:
        for _, r in pr_df.head(8).iterrows():
            color = "#ef4444" if r["score"] >= 0.7 else ("#f97316" if r["score"] >= 0.45 else "#22c55e")
            label = "🔴 CRITICAL" if r["score"] >= 0.7 else ("🟠 HIGH" if r["score"] >= 0.45 else "🟢 NORMAL")
            status_color = stage_colors.get(r["status"], "#ffffff")
            sla_html = ("<span style='color:#ef4444;'>❗ OVERDUE</span>" if r["overdue"] else f"<span style='color:#2563eb;'>⏳ {r['time_left']}h left</span>")
            conv_html = ""
            if r["prob"] is not None:
                pct = int(r["prob"] * 100)
                conv_color = "#22c55e" if pct > 70 else ("#f97316" if pct > 40 else "#ef4444")
                conv_html = f"<span style='margin-left:10px;color:{conv_color};'>📊 {pct}% Win</span>"

            st.markdown(f"""
            <div class="priority-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="margin-bottom:6px;"><b style="color:{color};">{label}</b>
                           <span class="stage-badge" style="background:{status_color}20;color:{status_color};border:1px solid {status_color}40;">{r['status']}</span>
                        </div>
                        <div style="font-size:16px;font-weight:700;">#{r['id']} — {r['name']}</div>
                        <div style="color:#93a0ad;font-size:13px;">{r['damage'].title()} | Est: <span style="color:#22c55e;font-weight:700;">{format_currency(r['value'])}</span></div>
                        <div style="margin-top:6px;font-size:13px;">{sla_html} {conv_html}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:32px;font-weight:700;color:{color};">{r['score']:.2f}</div>
                        <div style="font-size:11px;color:#93a0ad;">Priority</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- Expandable lead cards with full edit functionality (keeps original behavior)
    st.markdown("### 📋 All Leads (expand to edit)")
    # show filtered leads (if filter applied) else all
    leads_to_show = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if st.session_state.pipeline_stage_filter:
        leads_to_show = [L for L in leads_to_show if L.status == st.session_state.pipeline_stage_filter]

    for lead in leads_to_show:
        est_val = lead.estimated_value or 0.0
        card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {format_currency(est_val)}"
        with st.expander(card_title, expanded=False):
            colA, colB = st.columns([3,1])
            with colA:
                st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                st.markdown(f"**Address:** {lead.property_address or '—'}")
                st.markdown(f"**Notes:** {lead.notes or '—'}")
                st.markdown(f"**Created:** {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}")
            with colB:
                entered = lead.sla_entered_at or lead.created_at
                if isinstance(entered, str):
                    try: entered = datetime.fromisoformat(entered)
                    except: entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    sla_status_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                else:
                    hours = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    sla_status_html = f"<div style='color:#2563eb;font-weight:600;'>⏳ {hours}h {mins}m</div>"
                st.markdown(f"<div style='text-align:right'><div class='stage-badge' style='background:{stage_colors.get(lead.status,'#fff')}20;color:{stage_colors.get(lead.status,'#fff')};border:1px solid {stage_colors.get(lead.status,'#fff')}40;'>{lead.status}</div>{sla_status_html}</div>", unsafe_allow_html=True)

            st.markdown("---")
            # Quick contact buttons
            qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
            phone = (lead.contact_phone or "").strip()
            email = (lead.contact_email or "").strip()
            if phone:
                qc1.markdown(f"<a href='tel:{phone}'><button class='quick-call'>📞 Call</button></a>", unsafe_allow_html=True)
                wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                qc2.markdown(f"<a href='{wa_link}' target='_blank'><button class='quick-wa'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
            else:
                qc1.write(" "); qc2.write(" ")
            if email:
                qc3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='background:transparent; color:var(--white); border:1px solid rgba(255,255,255,0.12); border-radius:8px; padding:6px 10px;'>✉️ Email</button></a>", unsafe_allow_html=True)
            else:
                qc3.write(" ")

            st.markdown("---")

            # Lead update form (safe commit + rerun)
            form_key = f"update_lead_{lead.id}"
            with st.form(form_key):
                ucol1, ucol2 = st.columns(2)
                with ucol1:
                    new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"status_{lead.id}")
                    new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                    contacted = st.checkbox("Contacted", value=lead.contacted, key=f"contacted_{lead.id}")
                with ucol2:
                    inspection_scheduled = st.checkbox("Inspection Scheduled", value=lead.inspection_scheduled, key=f"insp_sched_{lead.id}")
                    inspection_completed = st.checkbox("Inspection Completed", value=lead.inspection_completed, key=f"insp_comp_{lead.id}")
                    estimate_submitted = st.checkbox("Estimate Submitted", value=lead.estimate_submitted, key=f"est_sub_{lead.id}")
                new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                if st.form_submit_button("💾 Update Lead"):
                    s_db = get_session()
                    lead_db = s_db.query(Lead).filter(Lead.id == lead.id).first()
                    if lead_db:
                        lead_db.status = new_status
                        lead_db.assigned_to = new_assigned
                        lead_db.contacted = contacted
                        lead_db.inspection_scheduled = inspection_scheduled
                        lead_db.inspection_completed = inspection_completed
                        lead_db.estimate_submitted = estimate_submitted
                        lead_db.notes = new_notes
                        # update sla_stage/entered when status changes significantly
                        lead_db.sla_stage = new_status
                        lead_db.sla_entered_at = lead_db.sla_entered_at or datetime.utcnow()
                        s_db.add(lead_db)
                        s_db.commit()
                    st.success(f"Lead #{lead.id} updated!")
                    # safe rerun to refresh UI
                    st.experimental_rerun()

            st.markdown("---")
            # Estimates
            lead_estimates = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
            if lead_estimates:
                for est in lead_estimates:
                    est_status = "✅ Approved" if est.approved else ("❌ Lost" if est.lost else "⏳ Pending")
                    est_color = "#22c55e" if est.approved else ("#ef4444" if est.lost else "#f97316")
                    st.markdown(f"<div style='padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;margin:8px 0;'><div style='display:flex;justify-content:space-between;align-items:center;'><div><span style='color:{est_color};font-weight:700;'>{est_status}</span><span style='margin-left:12px;color:#22c55e;font-weight:700;font-size:18px;'>{format_currency(est.amount)}</span></div><div style='color:#93a0ad;font-size:12px;'>{est.created_at.strftime('%Y-%m-%d') if est.created_at else '—'}</div></div></div>", unsafe_allow_html=True)
            else:
                st.info("No estimates yet for this lead.")

            # Create estimate form
            with st.form(f"create_estimate_{lead.id}"):
                st.markdown("**Create New Estimate**")
                est_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                if st.form_submit_button("➕ Create Estimate"):
                    create_estimate(get_session(), lead.id, est_amount, est_details)
                    st.success("Estimate created!")
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
            sla_entered_at = row.get("sla_entered_at")
            try:
                if pd.isna(sla_entered_at) or sla_entered_at is None:
                    sla_entered_at = datetime.utcnow()
                elif isinstance(sla_entered_at, str):
                    sla_entered_at = datetime.fromisoformat(sla_entered_at)
            except Exception:
                sla_entered_at = datetime.utcnow()
            sla_hours = int(row.get("sla_hours") or 24)
            deadline = sla_entered_at + timedelta(hours=sla_hours)
            remaining = deadline - datetime.utcnow()
            overdue_rows.append({
                "id": row["id"],
                "contact": row.get("contact_name"),
                "status": row.get("status"),
                "sla_stage": row.get("sla_stage"),
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
