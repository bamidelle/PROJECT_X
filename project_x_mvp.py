# project_x_mvp_fixed_full.py
"""
Project X — Full single-file app (Fixed)
- Lead Capture (qualified yes/no)
- Pipeline Board (editable rows), Estimates, Awarded invoice upload
- SLA engine, Priority scoring, Analytics, Exports
- Styling: Roboto, white labels, black user-entered text,
  select hover black, main submission buttons red with black text,
  quick contact buttons colored (WhatsApp green, Call blue)
"""

import os
from datetime import datetime, timedelta, time as dtime, date as dt_date
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# CONFIG
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_mvp_fixed_full.db")
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
    "qualified": "INTEGER DEFAULT 0",
}

# ---------------------------
# CSS / UI
# ---------------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

:root{
  --bg:#0b0f13;
  --card:#0f1720;
  --muted:#93a0ad;
  --white:#ffffff;
  --placeholder:#3a3a3a;
  --radius:10px;
  --primary-red:#ff2d2d;
  --money-green:#22c55e;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}

/* base */
body, .stApp {
  background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%);
  color: var(--white);
  font-family: 'Roboto', sans-serif;
}

/* sidebar */
section[data-testid="stSidebar"] {
  background: transparent !important;
  padding: 18px;
  border-right: 1px solid rgba(255,255,255,0.03);
}

/* header */
.header {
  padding: 12px;
  border-radius: 8px;
  color: var(--white);
  font-weight: 600;
  font-size: 18px;
}

/* form titles and control labels */
h1, h2, h3, label, .css-1rs6os { color: var(--white) !important; }

/* make labels and general text white */
div, p, span, label {
  color: var(--white) !important;
}

/* placeholder color */
input::placeholder, textarea::placeholder {
  color: var(--placeholder) !important;
}

/* form inputs: transparent background, deep black typed text */
input, textarea, select {
  background: rgba(255,255,255,0.01) !important;
  color: #000000 !important; /* deep black for user-entered text */
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* date/time pickers */
input[type="date"], input[type="time"] {
  color: #000000 !important;
}

/* default Streamlit button style (keeps prior look) */
button.stButton > button, .stButton>button {
  background: transparent !important;
  color: var(--white) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* MAIN submission buttons (forms) - style to red with black text */
button.stButton[data-testid="stFormSubmitButton"] > button,
div[data-testid="stFormSubmitButton"] > button,
.stButton>button[data-testid="stFormSubmitButton"] {
  background: var(--primary-red) !important;
  color: #000000 !important;
  border: 1px solid var(--primary-red) !important;
}

/* Quick-contact button inline styles will set colors; fallback class */
.btn-call { background: var(--call-blue); color: #000; border-radius:8px; padding:6px 10px; border:none; }
.btn-wa   { background: var(--wa-green); color: #000; border-radius:8px; padding:6px 10px; border:none; }

/* Money values (green) */
.money { color: var(--money-green); font-weight:700; }

/* Select option hover (browser-dependent) */
select option:hover { background: #000000 !important; color: #ffffff !important; }

/* small kv */
.kv { color: var(--white); font-size:13px; }

/* Funnel palette helpers */
.funnel-color-0 { color: #2563eb } /* blue */
.funnel-color-1 { color: #ffd2b3 } /* peach */
.funnel-color-2 { color: #22c55e } /* green */
.funnel-color-3 { color: #facc15 } /* yellow */
.funnel-color-4 { color: #fb923c } /* orange */
.funnel-color-5 { color: #000000 } /* black */
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
                # ignore migration failures (SQLite quirks)
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
    # update lead immediately
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
def combine_date_time(d: dt_date, t: dtime):
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
st.set_page_config(page_title="Project X — MVP Fixed", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker (Fixed)</div>", unsafe_allow_html=True)

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
    st.markdown('<small class="kv">Tip: Increase SLA weight to prioritise leads nearing deadline; increase value weight to prioritise larger jobs.</small>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(
            s,
            source="Google Ads",
            source_details="gclid=demo",
            contact_name="Demo Customer",
            contact_phone="+15550000",
            contact_email="demo@example.com",
            property_address="100 Demo Ave",
            damage_type="water",
            assigned_to="Alex",
            estimated_value=None,
            notes="Demo lead",
            sla_hours=24,
            qualified=True
        )
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
            st.experimental_rerun()

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

        # Priority summary top
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
                html = f"""
                <div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:space-between;'>
                  <div>
                    <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                    <span style='color:var(--muted); margin-left:8px;'>| Est: <span class="money">${r['estimated_value']:,.0f}</span></span>
                    <span style='color:var(--muted); margin-left:8px;'>| Time left: {int(r['time_left_hours'])}h</span>
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
            est_val_display = f"<span class='money'>${lead.estimated_value:,.0f}</span>" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'No damage type'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                    st.markdown(f"**Qualified:** {'Yes' if lead.qualified else 'No'}")
                with colB:
                    try:
                        single_row = df[df["id"] == lead.id].iloc[0].to_dict()
                        score, _, _, _, _, _, time_left = compute_priority_for_lead_row(single_row, weights)
                    except Exception:
                        score = 0.0; time_left = 9999
                    priority_label = ("High" if score >= 0.7 else "Medium" if score >= 0.45 else "Normal")
                    priority_color = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                    st.markdown(f"<div style='text-align:right'><strong style='color:{priority_color};'>{priority_label}</strong><br><span style='color:var(--muted);'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact (Call blue, WhatsApp green, Email default)
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    qc1.markdown(f"<a href='tel:{phone}'><button class='btn-call'>📞 Call</button></a>", unsafe_allow_html=True)
                    wa_number = phone.lstrip("+").replace(" ", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-wa'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")
                if email:
                    qc3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")

                # SLA
                entered = lead.sla_entered_at or lead.created_at
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except Exception:
                        entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    st.markdown(f"❗ <strong style='color:red;'>SLA OVERDUE</strong> — was due {deadline.strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                st.markdown("---")

                # Editable form (single lead)
                form_key = f"edit_lead_{lead.id}"
                with st.form(form_key):
                    c1, c2 = st.columns(2)
                    with c1:
                        contact_name = st.text_input("Contact name", value=lead.contact_name or "", key=f"cname_{lead.id}")
                        contact_phone = st.text_input("Contact phone", value=lead.contact_phone or "", key=f"cphone_{lead.id}")
                        contact_email = st.text_input("Contact email", value=lead.contact_email or "", key=f"cemail_{lead.id}")
                        property_address = st.text_input("Property address", value=lead.property_address or "", key=f"addr_{lead.id}")
                        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"], index=(["water","fire","mold","contents","reconstruction","other"].index(lead.damage_type) if lead.damage_type in ["water","fire","mold","contents","reconstruction","other"] else 5), key=f"damage_{lead.id}")
                    with c2:
                        assigned_to = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        est_val_display = lead.estimated_value or 0.0
                        # show estimated value widget (it will only be applied when Estimate Submitted=Yes)
                        est_val_widget = st.number_input("Estimated value (USD) — (applies if Estimate Submitted = Yes)", min_value=0.0, value=float(est_val_display), step=50.0, key=f"est_{lead.id}")
                        sla_hours = st.number_input("SLA hours", min_value=1, value=int(lead.sla_hours or 24), step=1, key=f"sla_{lead.id}")
                        status_choice = st.selectbox("Status", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")

                    notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")

                    st.markdown("**Pipeline Steps**")
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        contacted_choice = st.selectbox("Contacted?", ["No", "Yes"], index=1 if lead.contacted else 0, key=f"cont_{lead.id}")
                        inspection_scheduled_choice = st.selectbox("Inspection Scheduled?", ["No", "Yes"], index=1 if lead.inspection_scheduled else 0, key=f"inspsch_{lead.id}")
                        if inspection_scheduled_choice == "Yes":
                            default_date = lead.inspection_scheduled_at.date() if lead.inspection_scheduled_at else datetime.utcnow().date()
                            default_time = (lead.inspection_scheduled_at.time() if lead.inspection_scheduled_at else dtime(hour=9, minute=0))
                            insp_date = st.date_input("Inspection date", value=default_date, key=f"insp_date_{lead.id}")
                            insp_time = st.time_input("Inspection time", value=default_time, key=f"insp_time_{lead.id}")
                            inspection_dt = combine_date_time(insp_date, insp_time)
                        else:
                            inspection_dt = None
                    with f2:
                        inspection_completed_choice = st.selectbox("Inspection Completed?", ["No","Yes"], index=1 if lead.inspection_completed else 0, key=f"inspcomp_{lead.id}")
                        if inspection_completed_choice == "Yes":
                            default_date2 = lead.inspection_completed_at.date() if lead.inspection_completed_at else datetime.utcnow().date()
                            default_time2 = (lead.inspection_completed_at.time() if lead.inspection_completed_at else dtime(hour=9, minute=0))
                            comp_date = st.date_input("Inspection completed date", value=default_date2, key=f"insp_comp_date_{lead.id}")
                            comp_time = st.time_input("Inspection completed time", value=default_time2, key=f"insp_comp_time_{lead.id}")
                            inspection_comp_dt = combine_date_time(comp_date, comp_time)
                        else:
                            inspection_comp_dt = None
                        estimate_sub_choice = st.selectbox("Estimate Submitted?", ["No","Yes"], index=1 if lead.estimate_submitted else 0, key=f"estsub_{lead.id}")
                        if estimate_sub_choice == "Yes":
                            est_submitted_date = st.date_input("Estimate submitted date", value=(lead.estimate_submitted_at.date() if lead.estimate_submitted_at else datetime.utcnow().date()), key=f"est_sub_date_{lead.id}")
                            est_submitted_time = st.time_input("Estimate submitted time", value=(lead.estimate_submitted_at.time() if lead.estimate_submitted_at else dtime(hour=9, minute=0)), key=f"est_sub_time_{lead.id}")
                            est_submitted_dt = combine_date_time(est_submitted_date, est_submitted_time)
                            est_amount_input = est_val_widget
                        else:
                            est_submitted_dt = None
                            est_amount_input = None
                    with f3:
                        awarded_choice = st.selectbox("Awarded?", ["No","Yes"], index=1 if lead.status == LeadStatus.AWARDED else 0, key=f"awarded_choice_{lead.id}")
                        awarded_comment = st.text_input("Awarded comment (optional)", value=lead.awarded_comment or "", key=f"awcom_{lead.id}")
                        awarded_date_val = lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()
                        awarded_date = st.date_input("Awarded date (optional)", value=awarded_date_val, key=f"awdate_{lead.id}")
                        awarded_invoice_upload = None
                        if awarded_choice == "Yes":
                            awarded_invoice_upload = st.file_uploader("Upload invoice (optional)", type=["pdf","jpg","png","jpeg"], key=f"inv_{lead.id}")
                        lost_choice = st.selectbox("Lost?", ["No","Yes"], index=1 if lead.status == LeadStatus.LOST else 0, key=f"lost_choice_{lead.id}")
                        lost_comment = st.text_input("Lost comment (optional)", value=lead.lost_comment or "", key=f"lostcom_{lead.id}")
                        lost_date_val = lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()
                        lost_date = st.date_input("Lost date (optional)", value=lost_date_val, key=f"lostdate_{lead.id}")

                    save = st.form_submit_button("Save Lead")
                    if save:
                        try:
                            lead.contact_name = contact_name.strip() or None
                            lead.contact_phone = contact_phone.strip() or None
                            lead.contact_email = contact_email.strip() or None
                            lead.property_address = property_address.strip() or None
                            lead.damage_type = damage_type
                            lead.assigned_to = assigned_to.strip() or None
                            if estimate_sub_choice == "Yes" and est_amount_input is not None:
                                lead.estimated_value = float(est_amount_input)
                                lead.estimate_submitted = True
                                lead.estimate_submitted_at = est_submitted_dt or datetime.utcnow()
                                lead.status = LeadStatus.ESTIMATE_SUBMITTED
                            lead.notes = notes.strip() or None
                            lead.sla_hours = int(sla_hours)
                            lead.contacted = True if contacted_choice == "Yes" else False
                            lead.inspection_scheduled = True if inspection_scheduled_choice == "Yes" else False
                            lead.inspection_scheduled_at = inspection_dt
                            lead.inspection_completed = True if inspection_completed_choice == "Yes" else False
                            lead.inspection_completed_at = inspection_comp_dt
                            if awarded_choice == "Yes":
                                lead.status = LeadStatus.AWARDED
                                lead.awarded_comment = awarded_comment.strip() or None
                                lead.awarded_date = datetime.combine(awarded_date, datetime.min.time()) if awarded_date else None
                                if awarded_invoice_upload is not None:
                                    saved_path = save_uploaded_file(awarded_invoice_upload, lead.id)
                                    if saved_path:
                                        lead.awarded_invoice = saved_path
                            else:
                                # don't remove previous awarded values automatically
                                pass
                            if lost_choice == "Yes":
                                lead.status = LeadStatus.LOST
                                lead.lost_comment = lost_comment.strip() or None
                                lead.lost_date = datetime.combine(lost_date, datetime.min.time()) if lost_date else None
                            if status_choice != lead.status and status_choice not in (LeadStatus.AWARDED, LeadStatus.LOST):
                                lead.status = status_choice
                                lead.sla_stage = status_choice
                                lead.sla_entered_at = datetime.utcnow()
                            s.add(lead)
                            s.commit()
                            st.success(f"Lead #{lead.id} saved.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error saving lead: {e}")

                # Estimates section
                st.markdown("**Estimates**")
                ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if ests:
                    est_rows = []
                    for e in ests:
                        est_rows.append({
                            "id": e.id,
                            "amount": e.amount,
                            "sent_at": e.sent_at,
                            "approved": e.approved,
                            "lost": e.lost,
                            "lost_reason": e.lost_reason,
                            "created_at": e.created_at
                        })
                    st.dataframe(pd.DataFrame(est_rows))
                    first_est = ests[0]
                    ea, eb, ec = st.columns(3)
                    with ea:
                        if st.button(f"Mark Sent (#{first_est.id})", key=f"send_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_sent(s, first_est.id)
                                st.success("Marked as sent.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(e)
                    with eb:
                        if st.button(f"Mark Approved (#{first_est.id})", key=f"app_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_approved(s, first_est.id)
                                st.success("Approved and lead moved to Awarded.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(e)
                    with ec:
                        if st.button(f"Mark Lost (#{first_est.id})", key=f"lost_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_lost(s, first_est.id, reason="Lost to competitor")
                                st.success("Marked lost and lead moved to Lost.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(e)
                else:
                    st.write("No estimates yet.")
                    with st.form(f"create_est_{lead.id}", clear_on_submit=True):
                        amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"new_est_amt_{lead.id}")
                        det = st.text_area("Estimate details (optional)", key=f"new_est_det_{lead.id}")
                        create_btn = st.form_submit_button("Create Estimate")
                        if create_btn:
                            try:
                                create_estimate(s, lead.id, float(amt), details=det)
                                st.success("Estimate created.")
                                st.experimental_rerun()
                            except Exception as e:
                                st.error(e)

                st.markdown("---")

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
        fig = px.bar(funnel, x="stage", y="count", title="Leads by Stage", text="count", color="stage", color_discrete_sequence=colors[: len(funnel)])
        fig.update_layout(xaxis_title=None, yaxis_title="Number of Leads", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Summary")
        total_leads = len(df)
        awarded = len(df[df.status == LeadStatus.AWARDED])
        lost = len(df[df.status == LeadStatus.LOST])
        contacted_cnt = int(df.contacted.sum()) if "contacted" in df.columns else 0
        insp_sched_cnt = int(df.inspection_scheduled.sum()) if "inspection_scheduled" in df.columns else 0
        st.markdown(f"- Total leads: **{total_leads}**")
        st.markdown(f"- Awarded: **{awarded}**")
        st.markdown(f"- Lost: **{lost}**")
        st.markdown(f"- Contacted: **{contacted_cnt}**")
        st.markdown(f"- Inspections scheduled: **{insp_sched_cnt}**")

        st.subheader("Conversion by Source")
        conv = df.copy()
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(leads=("id", "count"), awarded=("awarded_flag", "sum")).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        st.subheader("Qualified vs Unqualified — Time Breakdown")
        ts = df.copy()
        ts["created_date"] = pd.to_datetime(ts["created_at"]).dt.date
        ts["qualified_flag"] = ts["qualified"].apply(lambda x: 1 if x else 0) if "qualified" in ts.columns else 0
        choice = st.selectbox("Range", ["Daily", "Weekly", "Monthly", "Yearly"])
        if choice == "Daily":
            agg = ts.groupby("created_date").agg(total=("id", "count"), qualified=("qualified_flag", "sum")).reset_index()
            agg["unqualified"] = agg["total"] - agg["qualified"]
            x = "created_date"
        elif choice == "Weekly":
            ts["week"] = pd.to_datetime(ts["created_at"]).dt.to_period("W").apply(lambda r: r.start_time.date())
            agg = ts.groupby("week").agg(total=("id", "count"), qualified=("qualified_flag", "sum")).reset_index()
            agg["unqualified"] = agg["total"] - agg["qualified"]
            x = "week"
        elif choice == "Monthly":
            ts["month"] = pd.to_datetime(ts["created_at"]).dt.to_period("M").apply(lambda r: r.start_time.date())
            agg = ts.groupby("month").agg(total=("id", "count"), qualified=("qualified_flag", "sum")).reset_index()
            agg["unqualified"] = agg["total"] - agg["qualified"]
            x = "month"
        else:
            ts["year"] = pd.to_datetime(ts["created_at"]).dt.to_period("Y").apply(lambda r: r.start_time.date())
            agg = ts.groupby("year").agg(total=("id", "count"), qualified=("qualified_flag", "sum")).reset_index()
            agg["unqualified"] = agg["total"] - agg["qualified"]
            x = "year"

        if not agg.empty:
            fig2 = px.bar(agg, x=x, y=["qualified", "unqualified"], title=f"Qualified vs Unqualified ({choice})")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data for selected range.")

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
    st.markdown("---")
    st.write("Uploaded invoices (saved locally):")
    inv_dir = os.path.join(os.getcwd(), "uploaded_invoices")
    if os.path.isdir(inv_dir):
        files = os.listdir(inv_dir)
        if files:
            for f in files:
                st.write(f"- {f}")
        else:
            st.write("No uploaded invoices yet.")
    else:
        st.write("No uploaded invoices yet.")

# End of file
