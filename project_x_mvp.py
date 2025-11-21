# project_x_projectx_v1_full.py
"""
Project X — Complete MVP (Steps 1, 2, 3) — Full integrated app

Features implemented:
- Lead Capture (Is the Lead Qualified? instead of Estimated value)
- Pipeline Board (rows, fully editable). Estimated value is shown only when Estimate Submitted == Yes
- Awarded: Yes/No with comment, date, and invoice upload when Yes
- SLA engine: deadline, overdue detection
- Priority scoring (tunable weights)
- Analytics: Funnel (by stage) + Qualified vs Unqualified (daily/weekly/monthly/yearly)
- Safe in-app migration for added columns
- Styling as requested (Roboto, labels white, inputs black, form-submit buttons red/black,
  WhatsApp green, Call blue, money values green, dropdown hover black)
"""

import os
from datetime import datetime, timedelta, time
import pathlib
import io

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
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_projectx_v1_full.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# MIGRATION COLUMNS (SQLite text/integer)
# ---------------------------
MIGRATION_COLUMNS = {
    "contacted": "INTEGER DEFAULT 0",
    "inspection_scheduled": "INTEGER DEFAULT 0",
    "inspection_scheduled_at": "TEXT",
    "inspection_completed": "INTEGER DEFAULT 0",
    "inspection_completed_at": "TEXT",
    "estimate_submitted": "INTEGER DEFAULT 0",
    "estimate_submitted_at": "TEXT",
    "estimated_value": "REAL",  # moved to pipeline
    "awarded": "INTEGER DEFAULT 0",  # 0/1 for Yes/No
    "awarded_comment": "TEXT",
    "awarded_date": "TEXT",
    "award_invoice": "TEXT",
    "lost_comment": "TEXT",
    "lost_date": "TEXT",
    "qualified": "INTEGER DEFAULT 0",  # yes/no from lead capture
}

# ---------------------------
# CSS / UI
# ---------------------------
APP_CSS = r'''
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#0b0f13;
  --muted:#93a0ad;
  --white:#ffffff;
  --placeholder:#3a3a3a;
  --radius:10px;
  --primary:#ff2d2d; /* red for main submit buttons */
  --wa:#25D366; /* whatsapp green */
  --call:#1E90FF; /* call blue */
  --money:#2ecc71; /* green money */
}

/* base */
body, .stApp {
  background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%);
  color: var(--white);
  font-family: 'Roboto', sans-serif;
}

/* header */
.header { padding: 12px; color: var(--white); font-weight:600; font-size:18px; }

/* form inputs: transparent background, deep black typed text */
input, textarea, select {
  background: rgba(255,255,255,0.01) !important;
  color: #000000 !important; /* deep black for user-entered text */
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* date/time pickers */
input[type="datetime-local"], input[type="date"], input[type="time"] {
  color: #000000 !important;
}

/* placeholders */
input::placeholder, textarea::placeholder { color: var(--placeholder) !important; }

/* form submit buttons only: red background + black text */
/* Streamlit adds data-testid="stFormSubmitButton" to form submit buttons; target that */
button[data-testid="stFormSubmitButton"] {
  background: var(--primary) !important;
  color: #000000 !important;
  border: 1px solid var(--primary) !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
}

/* other buttons remain default; we'll style whatsapp/call inline */

/* money text */
.money { color: var(--money) !important; font-weight:700; }

/* small kv */
.kv { color: var(--muted); font-size:13px; }

/* select option hover - force black background and white text */
select option:hover { background: #000000 !important; color: #ffffff !important; }

/* For the dropdown overlay (some browsers) */
div[role="listbox"] > div[role="option"]:hover {
  background: #000000 !important;
  color: #ffffff !important;
}
'''

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

    # contact / property
    source = Column(String, default="Unknown")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)

    # capture: qualified flag (added migration)
    qualified = Column(Boolean, default=False)

    # pipeline
    status = Column(String, default=LeadStatus.NEW)
    assigned_to = Column(String, nullable=True)
    estimated_value = Column(Float, nullable=True)  # moved to pipeline
    notes = Column(Text, nullable=True)

    # SLA
    sla_hours = Column(Integer, default=24)
    sla_stage = Column(String, default=LeadStatus.NEW)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)

    # pipeline extras
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)

    awarded = Column(Boolean, default=False)
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    award_invoice = Column(Text, nullable=True)

    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)

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
# DB init + migration helpers
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
                # don't fail on single column issues
                print("Migration add column failed:", col, e)
    conn.close()

def init_db():
    create_tables_and_migrate()

# ---------------------------
# DB helpers
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
        qualified=kwargs.get("qualified", False),
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
        awarded=kwargs.get("awarded", False),
        awarded_comment=kwargs.get("awarded_comment"),
        awarded_date=kwargs.get("awarded_date"),
        award_invoice=kwargs.get("award_invoice"),
        lost_comment=kwargs.get("lost_comment"),
        lost_date=kwargs.get("lost_date"),
    )
    session.add(lead)
    session.commit()
    return lead

def leads_df(session):
    return pd.read_sql(session.query(Lead).statement, session.bind)

def estimates_df(session):
    return pd.read_sql(session.query(Estimate).statement, session.bind)

def create_estimate(session, lead_id, amount, details=""):
    est = Estimate(lead_id=lead_id, amount=amount, details=details)
    session.add(est)
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
        lead.awarded = True
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
    total_weight = (weights.get("value_weight", 0.5) + weights.get("sla_weight", 0.35) + weights.get("urgency_weight", 0.15))
    if total_weight <= 0:
        total_weight = 1.0

    score = (value_score * weights.get("value_weight", 0.5)
            + sla_score * weights.get("sla_weight", 0.35)
            + urgency_component * weights.get("urgency_weight", 0.15)) / total_weight
    score = max(0.0, min(score, 1.0))
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Project X — Full", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()

st.markdown("<div class='header'>Project X — Sales & Conversion Tracker (Full)</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

    # priority weights in session
    if "weights" not in st.session_state:
        st.session_state["weights"] = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5,
            "value_baseline": 5000.0
        }
    st.markdown("### Priority weight tuning")
    st.session_state["weights"]["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(st.session_state["weights"]["value_weight"]), step=0.05)
    st.session_state["weights"]["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(st.session_state["weights"]["sla_weight"]), step=0.05)
    st.session_state["weights"]["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(st.session_state["weights"]["urgency_weight"]), step=0.05)
    st.markdown("Within urgency flags:")
    st.session_state["weights"]["contacted_w"] = st.slider("  Not-contacted weight", 0.0, 1.0, float(st.session_state["weights"]["contacted_w"]), step=0.05)
    st.session_state["weights"]["inspection_w"] = st.slider("  Not-scheduled weight", 0.0, 1.0, float(st.session_state["weights"]["inspection_w"]), step=0.05)
    st.session_state["weights"]["estimate_w"] = st.slider("  No-estimate weight", 0.0, 1.0, float(st.session_state["weights"]["estimate_w"]), step=0.05)
    st.session_state["weights"]["value_baseline"] = st.number_input("Value baseline", min_value=100.0, value=float(st.session_state["weights"]["value_baseline"]), step=100.0)
    st.markdown('<small class="kv">Tip: Increase SLA weight to prioritise leads nearing deadline; increase value weight to prioritise larger jobs.</small>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", source_details="gclid=demo",
                 contact_name="Demo Customer", contact_phone="+15550000", contact_email="demo@example.com",
                 property_address="100 Demo Ave", damage_type="water",
                 qualified=True,
                 assigned_to="Alex", estimated_value=None, notes="Demo lead", sla_hours=24)
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
            # Removed Estimated value here — replaced by qualified selection
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context...")

        submitted = st.form_submit_button("Create Lead")
        if submitted:
            s = get_session()
            lead = add_lead(
                s,
                source=source, source_details=source_details,
                contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email,
                property_address=property_address, damage_type=damage_type,
                assigned_to=assigned_to,
                estimated_value=None,  # left blank until estimate submitted
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

# --- Page: Pipeline Board
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board — Rows (Fully editable)")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.get("weights", {"value_weight":0.5,"sla_weight":0.35,"urgency_weight":0.15,"contacted_w":0.6,"inspection_w":0.5,"estimate_w":0.5,"value_baseline":5000.0})

        # Priority top summary
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
                color = "red" if r["priority_score"] >= 0.7 else ("orange" if r["priority_score"] >= 0.45 else "white")
                st.markdown(
                    f"<div style='padding:8px;border-radius:8px;margin-bottom:6px;border:1px solid rgba(255,255,255,0.04);display:flex;justify-content:space-between;align-items:center;'>"
                    f"<div><strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>"
                    f"<span style='color:var(--muted); margin-left:8px;'>| Est: <span class=\"money\">${r['estimated_value']:,.0f}</span></span>"
                    f"<span style='color:var(--muted); margin-left:8px;'>| Time left: {int(r['time_left_hours'])}h</span></div>"
                    f"<div style='font-weight:700;color:{color};'>Priority: {r['priority_score']:.2f}</div></div>", unsafe_allow_html=True
                )
        else:
            st.info("No priority leads yet.")
        st.markdown("---")

        # Render each lead row
        for lead in leads:
            est_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or '—'} — {est_display}"
            with st.expander(title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                    st.markdown(f"**Qualified:** {'Yes' if lead.qualified else 'No'}")
                with colB:
                    # compute priority for this lead
                    try:
                        single_row = df[df["id"] == lead.id].iloc[0].to_dict()
                        score, _, _, _, _, _, time_left = compute_priority_for_lead_row(single_row, weights)
                    except Exception:
                        score = 0.0
                        time_left = 9999
                    label = "High" if score >= 0.7 else ("Medium" if score >= 0.45 else "Normal")
                    colr = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                    st.markdown(f"<div style='text-align:right'><strong style='color:{colr};'>{label}</strong><br><span style='color:var(--muted)'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact buttons (styled individually)
                q1, q2, q3, q4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    # Call - blue button inline style
                    q1.markdown(f"<a href='tel:{phone}'><button style='background:#1E90FF;color:#000;border-radius:8px;padding:6px 10px;border:1px solid #1E90FF;'>📞 Call</button></a>", unsafe_allow_html=True)
                    # WhatsApp - green button
                    wa_url = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    q2.markdown(f"<a href='{wa_url}' target='_blank'><button style='background:#25D366;color:#000;border-radius:8px;padding:6px 10px;border:1px solid #25D366;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    q1.write(" "); q2.write(" ")
                if email:
                    q3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='padding:6px 10px;border-radius:8px;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    q3.write(" ")
                q4.write("")

                # SLA countdown
                entered = lead.sla_entered_at or lead.created_at
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except Exception:
                        entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    st.markdown(f"❗ <strong style='color:red;'>SLA OVERDUE</strong> — due {deadline.strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                st.markdown("---")

                # Editable form for this lead
                with st.form(f"edit_lead_{lead.id}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        contact_name = st.text_input("Contact name", value=lead.contact_name or "", key=f"cname_{lead.id}")
                        contact_phone = st.text_input("Contact phone", value=lead.contact_phone or "", key=f"cphone_{lead.id}")
                        contact_email = st.text_input("Contact email", value=lead.contact_email or "", key=f"cemail_{lead.id}")
                        property_address = st.text_input("Property address", value=lead.property_address or "", key=f"addr_{lead.id}")
                        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"],
                                                   index=(["water","fire","mold","contents","reconstruction","other"].index(lead.damage_type) if lead.damage_type in ["water","fire","mold","contents","reconstruction","other"] else 5),
                                                   key=f"damage_{lead.id}")
                    with c2:
                        assigned_to = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        # Estimated value is shown/edited only if estimate_submitted == True
                        est_val_display = float(lead.estimated_value or 0.0)
                        # We'll render a number_input below conditionally
                        sla_hours = st.number_input("SLA hours", min_value=1, value=int(lead.sla_hours or 24), step=1, key=f"sla_{lead.id}")
                        status_choice = st.selectbox("Status", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")

                    notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")

                    st.markdown("**Pipeline Steps**")
                    f1, f2, f3 = st.columns(3)
                    with f1:
                        contacted_choice = st.selectbox("Contacted?", ["No","Yes"], index=1 if lead.contacted else 0, key=f"cont_{lead.id}")
                        inspection_scheduled_choice = st.selectbox("Inspection Scheduled?", ["No","Yes"], index=1 if lead.inspection_scheduled else 0, key=f"inspsch_{lead.id}")
                        if inspection_scheduled_choice == "Yes":
                            # date + time inputs
                            default_date = (lead.inspection_scheduled_at.date() if lead.inspection_scheduled_at else datetime.utcnow().date()) if isinstance(lead.inspection_scheduled_at, datetime) else (datetime.utcnow().date())
                            default_time = (lead.inspection_scheduled_at.time() if lead.inspection_scheduled_at else datetime.utcnow().time()) if isinstance(lead.inspection_scheduled_at, datetime) else (datetime.utcnow().time())
                            inspection_date = st.date_input("Inspection date", value=default_date, key=f"insp_date_{lead.id}")
                            inspection_time = st.time_input("Inspection time", value=default_time, key=f"insp_time_{lead.id}")
                            inspection_dt = datetime.combine(inspection_date, inspection_time)
                        else:
                            inspection_dt = None
                    with f2:
                        inspection_completed_choice = st.selectbox("Inspection Completed?", ["No","Yes"], index=1 if lead.inspection_completed else 0, key=f"inspcomp_{lead.id}")
                        if inspection_completed_choice == "Yes":
                            default_c_date = (lead.inspection_completed_at.date() if lead.inspection_completed_at else datetime.utcnow().date()) if isinstance(lead.inspection_completed_at, datetime) else (datetime.utcnow().date())
                            default_c_time = (lead.inspection_completed_at.time() if lead.inspection_completed_at else datetime.utcnow().time()) if isinstance(lead.inspection_completed_at, datetime) else (datetime.utcnow().time())
                            insp_comp_date = st.date_input("Inspection completed date", value=default_c_date, key=f"insp_comp_date_{lead.id}")
                            insp_comp_time = st.time_input("Inspection completed time", value=default_c_time, key=f"insp_comp_time_{lead.id}")
                            inspection_comp_dt = datetime.combine(insp_comp_date, insp_comp_time)
                        else:
                            inspection_comp_dt = None

                        estimate_sub_choice = st.selectbox("Estimate Submitted?", ["No","Yes"], index=1 if lead.estimate_submitted else 0, key=f"estsub_{lead.id}")
                        if estimate_sub_choice == "Yes":
                            default_est_date = (lead.estimate_submitted_at.date() if lead.estimate_submitted_at else datetime.utcnow().date()) if isinstance(lead.estimate_submitted_at, datetime) else datetime.utcnow().date()
                            est_sub_date = st.date_input("Estimate submitted date", value=default_est_date, key=f"est_sub_date_{lead.id}")
                            # Show estimated value input only here
                            est_val = st.number_input("Estimated value (USD)", min_value=0.0, value=float(lead.estimated_value or 0.0), step=50.0, key=f"est_val_{lead.id}")
                        else:
                            est_val = None
                            est_sub_date = None
                    with f3:
                        awarded_choice = st.selectbox("Awarded?", ["No", "Yes"], index=1 if lead.awarded else 0, key=f"awarded_{lead.id}")
                        awarded_comment = st.text_input("Awarded comment (optional)", value=lead.awarded_comment or "", key=f"awcom_{lead.id}")
                        if awarded_choice == "Yes":
                            awarded_date_default = (lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()) if isinstance(lead.awarded_date, datetime) else datetime.utcnow().date()
                            awarded_date = st.date_input("Awarded date", value=awarded_date_default, key=f"awdate_{lead.id}")
                            # File uploader for invoice
                            uploaded_file = st.file_uploader("Upload invoice (optional)", key=f"invoice_{lead.id}")
                        else:
                            awarded_date = None
                            uploaded_file = None

                        lost_comment = st.text_input("Lost comment (optional)", value=lead.lost_comment or "", key=f"lostcom_{lead.id}")
                        lost_date = st.date_input("Lost date (optional)", value=(lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()), key=f"lostdate_{lead.id}")

                    save = st.form_submit_button("Save Lead")
                    if save:
                        try:
                            lead.contact_name = contact_name.strip() or None
                            lead.contact_phone = contact_phone.strip() or None
                            lead.contact_email = contact_email.strip() or None
                            lead.property_address = property_address.strip() or None
                            lead.damage_type = damage_type
                            lead.assigned_to = assigned_to.strip() or None
                            # Only persist estimated value if estimate_submitted == Yes
                            if estimate_sub_choice == "Yes" and est_val is not None:
                                lead.estimated_value = float(est_val)
                            # else keep existing estimated_value as-is (or None)
                            lead.notes = notes.strip() or None
                            lead.sla_hours = int(sla_hours)
                            lead.contacted = True if contacted_choice == "Yes" else False
                            lead.inspection_scheduled = True if inspection_scheduled_choice == "Yes" else False
                            lead.inspection_scheduled_at = inspection_dt
                            lead.inspection_completed = True if inspection_completed_choice == "Yes" else False
                            lead.inspection_completed_at = inspection_comp_dt
                            lead.estimate_submitted = True if estimate_sub_choice == "Yes" else False
                            lead.estimate_submitted_at = (datetime.combine(est_sub_date, time(0,0)) if est_sub_date else None)
                            # Awarded handling
                            lead.awarded = True if awarded_choice == "Yes" else False
                            lead.awarded_comment = awarded_comment.strip() or None
                            lead.awarded_date = (datetime.combine(awarded_date, time(0,0)) if awarded_date else None)
                            # Handle invoice upload save
                            if uploaded_file is not None:
                                # save file to uploads dir with lead id and timestamp
                                filename = f"lead_{lead.id}_invoice_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
                                safe_path = os.path.join(UPLOAD_DIR, filename)
                                with open(safe_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                lead.award_invoice = safe_path
                            # Lost fields
                            lead.lost_comment = lost_comment.strip() or None
                            lead.lost_date = (datetime.combine(lost_date, time(0,0)) if lost_date else None)
                            # status change
                            if status_choice != lead.status:
                                lead.status = status_choice
                                lead.sla_stage = status_choice
                                lead.sla_entered_at = datetime.utcnow()
                            s.add(lead)
                            s.commit()
                            st.success(f"Lead #{lead.id} saved.")
                        except Exception as e:
                            st.error(f"Error saving lead: {e}")

                # Estimates section (outside the edit form)
                st.markdown("**Estimates**")
                ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if ests:
                    est_rows = []
                    for e in ests:
                        est_rows.append({"id": e.id, "amount": e.amount, "sent_at": e.sent_at, "approved": e.approved, "lost": e.lost, "lost_reason": e.lost_reason, "created_at": e.created_at})
                    st.dataframe(pd.DataFrame(est_rows))
                    first = ests[0]
                    ea, eb, ec = st.columns(3)
                    with ea:
                        if st.button(f"Mark Sent (#{first.id})", key=f"send_{lead.id}_{first.id}"):
                            try:
                                mark_estimate_sent(s, first.id)
                                st.success("Marked sent")
                            except Exception as ex:
                                st.error(ex)
                    with eb:
                        if st.button(f"Mark Approved (#{first.id})", key=f"app_{lead.id}_{first.id}"):
                            try:
                                mark_estimate_approved(s, first.id)
                                st.success("Approved; lead Awarded")
                            except Exception as ex:
                                st.error(ex)
                    with ec:
                        if st.button(f"Mark Lost (#{first.id})", key=f"lost_{lead.id}_{first.id}"):
                            try:
                                mark_estimate_lost(s, first.id, reason="Lost to competitor")
                                st.success("Marked lost")
                            except Exception as ex:
                                st.error(ex)
                else:
                    st.write("No estimates yet.")
                    with st.form(f"create_est_{lead.id}", clear_on_submit=True):
                        new_amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"new_est_amt_{lead.id}")
                        det = st.text_area("Estimate details (optional)", key=f"new_est_det_{lead.id}")
                        create_btn = st.form_submit_button("Create Estimate")
                        if create_btn:
                            try:
                                create_estimate(s, lead.id, float(new_amt), details=det)
                                st.success("Estimate created")
                            except Exception as ex:
                                st.error(ex)

                st.markdown("---")

# --- Page: Analytics & SLA
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze.")
    else:
        # Funnel overview (counts by status)
        funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
        funnel.columns = ["stage", "count"]
        st.subheader("Funnel Overview")
        # color list requested (blue, peach, green, yellow, orange, black, others)
        colors = ["#1f77b4", "#ffcc99", "#2ecc71", "#f2d244", "#ff7f0e", "#000000", "#8c564b"]
        fig = px.bar(funnel, x="stage", y="count", title="Leads by Stage", text="count", color="stage", color_discrete_sequence=colors)
        fig.update_layout(xaxis_title=None, yaxis_title="Number of Leads", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        st.markdown("### Summary")
        total = len(df)
        awarded = len(df[df.status == LeadStatus.AWARDED])
        lost = len(df[df.status == LeadStatus.LOST])
        contacted_cnt = int(df.contacted.sum()) if "contacted" in df.columns else 0
        insp_cnt = int(df.inspection_scheduled.sum()) if "inspection_scheduled" in df.columns else 0
        estimate_cnt = int(df.estimate_submitted.sum()) if "estimate_submitted" in df.columns else 0
        st.markdown(f"- Total leads: **{total}**")
        st.markdown(f"- Awarded: **{awarded}**")
        st.markdown(f"- Lost: **{lost}**")
        st.markdown(f"- Contacted: **{contacted_cnt}**")
        st.markdown(f"- Inspections scheduled: **{insp_cnt}**")
        st.markdown(f"- Estimates submitted: **{estimate_cnt}**")

        # Conversion by source
        st.subheader("Conversion by Source")
        conv = df.copy()
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(leads=("id", "count"), awarded=("awarded_flag", "sum")).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        # SLA / Overdue leads
        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row["sla_entered_at"]
            try:
                if pd.isna(sla_entered_at):
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

        # Qualified vs Unqualified chart
        st.subheader("Qualified vs Unqualified (choose period)")
        # Prepare df with created_at as datetime
        df_q = df.copy()
        df_q["created_at"] = pd.to_datetime(df_q["created_at"])
        df_q["qualified_flag"] = df_q["qualified"].apply(lambda x: 1 if x else 0)
        period_option = st.selectbox("Period aggregation", ["Daily", "Weekly", "Monthly", "Yearly"], index=0)
        df_q.set_index("created_at", inplace=True)
        if period_option == "Daily":
            agg = df_q.resample("D").agg({"qualified_flag": "sum", "id": "count"}).rename(columns={"id": "total"})
        elif period_option == "Weekly":
            agg = df_q.resample("W").agg({"qualified_flag": "sum", "id": "count"}).rename(columns={"id": "total"})
        elif period_option == "Monthly":
            agg = df_q.resample("M").agg({"qualified_flag": "sum", "id": "count"}).rename(columns={"id": "total"})
        else:
            agg = df_q.resample("Y").agg({"qualified_flag": "sum", "id": "count"}).rename(columns={"id": "total"})
        if not agg.empty:
            agg = agg.reset_index()
            agg["unqualified"] = agg["total"] - agg["qualified_flag"]
            # melted for stacked bar
            m = agg.melt(id_vars=["created_at"], value_vars=["qualified_flag", "unqualified"], var_name="type", value_name="count")
            label_map = {"qualified_flag": "Qualified", "unqualified": "Unqualified"}
            m["type"] = m["type"].map(label_map)
            fig2 = px.bar(m, x="created_at", y="count", color="type", barmode="stack", title=f"Qualified vs Unqualified ({period_option})", text="count")
            fig2.update_layout(xaxis_title=None, yaxis_title="Leads", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough data for this aggregation period.")

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

# End of file
