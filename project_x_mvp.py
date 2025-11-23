# assan1_app.py
"""
Assan / Project X — Main app file (fixed)
- Single-file Streamlit app with SQLite + SQLAlchemy
- No experimental_rerun calls (avoids AttributeError)
- All datetime inputs use date_input + time_input (combine_date_time)
- Money values shown in green
- Main form submit buttons styled red with black text
- Quick contact buttons: Call (blue), WhatsApp (green), Email (transparent)
- Priority scoring, SLA countdown, Analytics (funnel + qualified breakdown)
"""

import os
from datetime import datetime, timedelta, time as dtime
import streamlit as st
import pandas as pd
import plotly.express as px

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# CONFIG
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
# CSS / UI
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

/* Form labels, control labels */
label, .css-1rs6os { color: var(--blue) !important; }

/* Inputs: black typed text */
input, textarea, select {
  background: rgba(255,255,255,0.01) !important;
  color: #000000 !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}
input::placeholder, textarea::placeholder { color: var(--placeholder) !important; }
input[type="date"], input[type="time"] { color: #000000 !important; }

/* Default streamlit button style kept */
button.stButton > button, .stButton>button {
  background: transparent !important;
  color: var(--white) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* Main form submission buttons -> red background with black text */
div[data-testid="stFormSubmitButton"] > button, button.stButton[data-testid="stFormSubmitButton"] > button {
  background: var(--primary-red) !important;
  color: #000000 !important;
  border: 1px solid var(--primary-red) !important;
}

/* Money values (green) */
.money { color: var(--money-green); font-weight:700; }

/* Select option hover - best effort (browser dependent) */
select option:hover { background: #000000 !important; color: #ffffff !important; }

/* Quick contact overrides for inline use */
.quick-call { background: var(--call-blue); color:#000; border-radius:8px; padding:6px 10px; border:none; }
.quick-wa { background: var(--wa-green); color:#000; border-radius:8px; padding:6px 10px; border:none; }

/* small help text */
.kv { color: var(--muted); font-size:13px; }
"""

# ---------------------------
# ORM MODELS
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
# DB init + safe migration
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
# DB helpers (CRUD)
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
        sla_entered_at=datetime.utcnow(),
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
# Priority scoring
# ---------------------------
def compute_priority_for_lead_row(lead_row, weights):
    val = float(lead_row.get("estimated_value") or 0.0)
    baseline = weights.get("value_baseline", 5000.0)
    value_score = min(val / baseline, 1.0)

    # time left hours
    try:
        sla_entered = lead_row.get("sla_entered_at")
        if pd.isna(sla_entered) or sla_entered is None:
            time_left_h = 9999.0
        else:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
            time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except Exception:
        time_left_h = 9999.0

    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)

    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0

    urgency_component = (contacted_flag * weights.get("contacted_w", 0.6)
                        + inspection_flag * weights.get("inspection_w", 0.5)
                        + estimate_flag * weights.get("estimate_w", 0.5))
    total_weight = (weights.get("value_weight", 0.5)
                   + weights.get("sla_weight", 0.35)
                   + weights.get("urgency_weight", 0.15))
    if total_weight <= 0:
        total_weight = 1.0
    score = (value_score * weights.get("value_weight", 0.5)
            + sla_score * weights.get("sla_weight", 0.35)
            + urgency_component * weights.get("urgency_weight", 0.15)) / total_weight
    score = max(0.0, min(score, 1.0))
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

# ---------------------------
# Utilities
# ---------------------------
def combine_date_time(d: datetime.date, t: dtime):
    if d is None and t is None:
        return None
    if d is None:
        d = datetime.utcnow().date()
    if t is None:
        t = dtime.min
    return datetime.combine(d, t)

def save_uploaded_file(uploaded_file, lead_id):
    if uploaded_file is None:
        return None
    folder = os.path.join(os.getcwd(), "uploaded_invoices")
    os.makedirs(folder, exist_ok=True)
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

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
            # replaced field
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
            # no st.experimental_rerun() - avoid attribute errors

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# --- Page: Pipeline Board
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board — Rows (editable)")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # Priority summary
        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left),
                "priority_score": score,
                "status": row.get("status"),
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        st.subheader("Priority Leads (Top 8)")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                color = "#ffffff"
                if score >= 0.7:
                    color = "red"
                elif score >= 0.45:
                    color = "orange"
                # estimated value inline green style to guarantee green display
                est_html = f"<span class='money' style='color:var(--money-green);'>${r['estimated_value']:,.0f}</span>"
                html = f"""
                <div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:space-between;'>
                  <div>
                    <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                    <span style='color:var(--muted); margin-left:8px;'>| Est: {est_html}</span>
                    <span style='color:var(--muted); margin-left:8px;'>| Time left: <span style="color:#2563eb;font-weight:700;">{int(r['time_left_hours'])}h</span></span>
                  </div>
                  <div style='font-weight:700;color:{color};'>Priority: {r['priority_score']:.2f}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
        else:
            st.info("No priority leads yet.")
        st.markdown("---")

        # Render rows
        for lead in leads:
            est_val_display = f"<span class='money' style='color:var(--money-green)'>${lead.estimated_value:,.0f}</span>" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'No damage type'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                with colB:
                    # compute single priority
                    try:
                        single_row = df[df["id"] == lead.id].iloc[0].to_dict()
                        score, _, _, _, _, _, time_left = compute_priority_for_lead_row(single_row, weights)
                    except Exception:
                        score = 0.0; time_left = 9999
                    priority_label = ("High" if score >= 0.7 else "Medium" if score >= 0.45 else "Normal")
                    priority_color = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                    # Time-left display in blue (static until next rerun)
                    time_left_display = f"{int(time_left)}h" if time_left < 9999 else "—"
                    st.markdown(f"<div style='text-align:right'><strong style='color:{priority_color};'>{priority_label}</strong><br><span style='color:#2563eb;font-weight:700;'>Time left: {time_left_display}</span></div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact (inline colored buttons)
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    qc1.markdown(f"<a href='tel:{phone}'><button class='quick-call'>📞 Call</button></a>", unsafe_allow_html=True)
                    wa_number = phone.lstrip("+").replace(" ", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button class='quick-wa'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")
                if email:
                    qc3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='background:transparent; color:var(--white); border:1px solid rgba(255,255,255,0.12); border-radius:8px; padding:6px 10px;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")

                import os
from datetime import datetime, timedelta, date as ddate, time as dtime
import pandas as pd
import streamlit as st
import plotly.express as px
from models import Lead, Estimate, LeadStatus, init_db, get_session, leads_df, estimates_df, add_lead, create_estimate, mark_estimate_sent, mark_estimate_approved, mark_estimate_lost
from utils import combine_date_time, save_uploaded_file, compute_priority_for_lead_row, predict_lead_probability
import joblib

# --- Streamlit Page Setup
st.set_page_config(page_title="Assan — CRM", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()
st.markdown("<div class='header'>Assan — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# --- Sidebar: Control + Priority tuning
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
    st.markdown("---")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# --- Page: Pipeline Board
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board — Rows (editable)")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # Load ML model if exists
        try:
            lead_model = joblib.load('lead_conversion_model.pkl')
        except:
            lead_model = None

        # Priority list with SLA & conversion probability
        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
            # SLA remaining
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try: sla_entered = datetime.fromisoformat(sla_entered)
                except: sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0
            # Predicted conversion
            prob = None
            if lead_model is not None:
                try: prob = predict_lead_probability(lead_model, row)
                except: prob = None
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "conversion_prob": prob
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        st.subheader("Priority Leads (Top 8)")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                color = "#ffffff"
                if score >= 0.7: color = "red"
                elif score >= 0.45: color = "orange"
                est_html = f"<span class='money' style='color:var(--money-green);'>${r['estimated_value']:,.0f}</span>"
                sla_html = f"❗ <strong style='color:red;'>OVERDUE (was due {r['sla_deadline'].strftime('%Y-%m-%d %H:%M')})</strong>" if r["sla_overdue"] else f"⏳ SLA remaining: <span style='color:#2563eb;font-weight:700;'>{int(r['time_left_hours']):02d}:{int(r['time_left_hours']*60%60):02d}</span> (due {r['sla_deadline'].strftime('%Y-%m-%d %H:%M')})"
                conv_html = f"<br><small>Predicted conversion probability: <strong>{r['conversion_prob']*100:.1f}%</strong></small>" if r["conversion_prob"] is not None else ""
                html = f"""
                <div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.04);display:flex;flex-direction:column;'>
                  <div><strong style='color:{color};'>#{int(r['id'])} — {r['contact_name']}</strong> | Est: {est_html} | Priority: {score:.2f}</div>
                  <div style='margin-top:4px;'>{sla_html}{conv_html}</div>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

        # Render individual lead cards (SLA + form + estimates)
        for lead in leads:
            est_val_display = f"<span class='money' style='color:var(--money-green)'>${lead.estimated_value:,.0f}</span>" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'No damage type'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3,1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                with colB:
                    # SLA + predicted conversion
                    entered = lead.sla_entered_at or lead.created_at
                    if isinstance(entered, str):
                        try: entered = datetime.fromisoformat(entered)
                        except: entered = datetime.utcnow()
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    sla_html = f"❗ <strong style='color:red;'>SLA OVERDUE</strong> — was due {deadline.strftime('%Y-%m-%d %H:%M')}" if remaining.total_seconds() <= 0 else f"⏳ SLA remaining: <span style='color:#2563eb;font-weight:700;'>{int(remaining.total_seconds()//3600):02d}:{int((remaining.total_seconds()%3600)//60):02d}</span> (due {deadline.strftime('%Y-%m-%d %H:%M')})"
                    prob = None
                    if lead_model is not None:
                        try: lead_row = df[df["id"]==lead.id].iloc[0].to_dict(); prob = predict_lead_probability(lead_model, lead_row)
                        except: prob = None
                    conv_html = f"<br><small>Predicted conversion probability: <strong>{prob*100:.1f}%</strong></small>" if prob is not None else ""
                    st.markdown(f"<div style='text-align:right'>{sla_html}{conv_html}</div>", unsafe_allow_html=True)

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
        colors = ["#2563eb", "#ffd2b3", "#22c55e", "#facc15", "#fb923c", "#000000", "#a3a3a3"]
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
