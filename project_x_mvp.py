# project_x_full.py
"""
Project X — Full single-file app
- Lead Capture (form)
- Pipeline Board (Clean Rows layout, fully editable)
- Analytics & SLA dashboard
- Exports (CSV)
- SQLite + SQLAlchemy ORM with light in-app migration for pipeline columns
- Priority scoring with tunable weights in sidebar
- Styling (Roboto, dark UI, white labels, black user input text)
"""

import os
from datetime import datetime, timedelta
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
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_full.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# pipeline columns for safe in-app migration (SQLite-compatible types)
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
    "lost_comment": "TEXT",
    "lost_date": "TEXT",
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
input[type="datetime-local"], input[type="date"], input[type="time"] {
  color: #000000 !important;
}

/* transparent button with white border & white text */
button.stButton > button {
  background: transparent !important;
  color: var(--white) !important;
  border: 1px solid var(--white) !important;
  padding: 8px 12px !important;
  border-radius: 8px !important;
  font-weight: 500 !important;
}

/* small kv */
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

    # contact / property
    source = Column(String, default="Unknown")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)

    # pipeline
    status = Column(String, default=LeadStatus.NEW)
    assigned_to = Column(String, nullable=True)
    estimated_value = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)

    # SLA
    sla_hours = Column(Integer, default=24)
    sla_stage = Column(String, default=LeadStatus.NEW)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)

    # pipeline extra fields (migration may add them)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
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
# DB init + safe migration
# ---------------------------
def create_tables_and_migrate():
    # create tables
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
                # continue on failure (SQLite quirks)
                print("Migration add column failed", col, e)
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
    # value normalized
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

    # SLA urgency 0..1
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)

    # flags
    contacted = bool(lead_row.get("contacted"))
    inspection_scheduled = bool(lead_row.get("inspection_scheduled"))
    estimate_submitted = bool(lead_row.get("estimate_submitted"))

    contacted_flag = 0.0 if contacted else 1.0
    inspection_flag = 0.0 if inspection_scheduled else 1.0
    estimate_flag = 0.0 if estimate_submitted else 1.0

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
# App UI
# ---------------------------
st.set_page_config(page_title="Project X — CRM", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

    # priority weight controls
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5,
            "value_baseline": 5000.0
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

    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", source_details="gclid=demo",
                 contact_name="Demo Customer", contact_phone="+15550000", contact_email="demo@example.com",
                 property_address="100 Demo Ave", damage_type="water",
                 assigned_to="Alex", estimated_value=4200.0, notes="Demo lead", sla_hours=24)
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
            estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=50.0)
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
                estimated_value=float(estimated_value) if estimated_value else None,
                notes=notes,
                sla_hours=int(sla_hours)
            )
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values('created_at', ascending=False).head(50))

# --- Page: Pipeline Board (Clean Rows layout — fully editable)
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board — Clean Rows")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        # Build DataFrame for priority scoring
        df = leads_df(s)
        weights = st.session_state.get("weights", {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        })

        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left_h = compute_priority_for_lead_row(row, weights)
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left_h),
                "priority_score": score,
                "status": row.get("status"),
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        # Priority summary at top
        st.subheader("Priority Leads (Top 8)")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                if score >= 0.7:
                    color = "red"
                elif score >= 0.45:
                    color = "orange"
                else:
                    color = "white"
                html_block = f"""
                <div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:space-between;'>
                    <div>
                        <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                        <span style='color:var(--muted); margin-left:8px;'>| Est: ${r['estimated_value']:,.0f}</span>
                        <span style='color:var(--muted); margin-left:8px;'>| Time left: {int(r['time_left_hours'])}h</span>
                    </div>
                    <div style='font-weight:600; color:{color};'>Priority: {r['priority_score']:.2f}</div>
                </div>
                """
                st.markdown(html_block, unsafe_allow_html=True)
        else:
            st.info("No priority leads yet.")
        st.markdown("---")

        # Render leads as vertical rows (clean cards)
        for lead in leads:
            # Header line for each lead (compact)
            est_val_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'No damage type'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                # Top info row
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                with col_b:
                    # compute and show priority for this specific lead
                    try:
                        single_row = df[df["id"] == lead.id].iloc[0].to_dict()
                        score, _, _, _, _, _, time_left_h = compute_priority_for_lead_row(single_row, weights)
                    except Exception:
                        score = 0.0
                        time_left_h = 9999
                    priority_label = ("High" if score >= 0.7 else "Medium" if score >= 0.45 else "Normal")
                    priority_color = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                    st.markdown(f"<div style='text-align:right'><strong style='color:{priority_color};'>{priority_label}</strong><br><span style='color:var(--muted);'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact actions (Call / WhatsApp / Email)
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    tel_link = f"tel:{phone}"
                    wa_link = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    qc1.markdown(f"<a href='{tel_link}'><button>📞 Call</button></a>", unsafe_allow_html=True)
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" ")
                    qc2.write(" ")
                if email:
                    mail_link = f"mailto:{email}?subject=Follow%20up%20on%20your%20restoration%20request"
                    qc3.markdown(f"<a href='{mail_link}'><button>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")  # spacer column

                # SLA countdown display
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

                # Editable form for this lead (NO st.rerun inside)
                form_key = f"lead_edit_form_{lead.id}"
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
                        est_val = st.number_input("Estimated value (USD)", min_value=0.0, value=float(lead.estimated_value or 0.0), step=50.0, key=f"est_{lead.id}")
                        sla_hours = st.number_input("SLA hours", min_value=1, value=int(lead.sla_hours or 24), step=1, key=f"sla_{lead.id}")
                        status_choice = st.selectbox("Status", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")

                    notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")

                    # Pipeline flags & dates
                    st.markdown("**Pipeline Steps**")
                    colf1, colf2, colf3 = st.columns(3)
                    with colf1:
                        contacted_choice = st.selectbox("Contacted?", ["No", "Yes"], index=1 if lead.contacted else 0, key=f"cont_{lead.id}")
                        inspection_scheduled_choice = st.selectbox("Inspection Scheduled?", ["No", "Yes"], index=1 if lead.inspection_scheduled else 0, key=f"inspsch_{lead.id}")
                        if inspection_scheduled_choice == "Yes":
                            default_dt = lead.inspection_scheduled_at or datetime.utcnow()
                            if isinstance(default_dt, str):
                                try:
                                    default_dt = datetime.fromisoformat(default_dt)
                                except Exception:
                                    default_dt = datetime.utcnow()
                            inspection_dt = st.datetime_input("Inspection date & time", value=default_dt, key=f"insp_dt_{lead.id}")
                        else:
                            inspection_dt = None
                    with colf2:
                        inspection_completed_choice = st.selectbox("Inspection Completed?", ["No", "Yes"], index=1 if lead.inspection_completed else 0, key=f"inspcomp_{lead.id}")
                        if inspection_completed_choice == "Yes":
                            default_dt2 = lead.inspection_completed_at or datetime.utcnow()
                            if isinstance(default_dt2, str):
                                try:
                                    default_dt2 = datetime.fromisoformat(default_dt2)
                                except Exception:
                                    default_dt2 = datetime.utcnow()
                            inspection_comp_dt = st.datetime_input("Inspection completed at", value=default_dt2, key=f"insp_comp_dt_{lead.id}")
                        else:
                            inspection_comp_dt = None
                        estimate_sub_choice = st.selectbox("Estimate Submitted?", ["No","Yes"], index=1 if lead.estimate_submitted else 0, key=f"estsub_{lead.id}")
                        if estimate_sub_choice == "Yes":
                            est_submitted_at = st.date_input("Estimate submitted at (optional)", value=(lead.estimate_submitted_at.date() if lead.estimate_submitted_at else datetime.utcnow().date()), key=f"est_sub_dt_{lead.id}")
                        else:
                            est_submitted_at = None
                    with colf3:
                        awarded_comment = st.text_input("Awarded comment (optional)", value=lead.awarded_comment or "", key=f"awcom_{lead.id}")
                        awarded_date = st.date_input("Awarded date (optional)", value=(lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()), key=f"awdate_{lead.id}")
                        lost_comment = st.text_input("Lost comment (optional)", value=lead.lost_comment or "", key=f"lostcom_{lead.id}")
                        lost_date = st.date_input("Lost date (optional)", value=(lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()), key=f"lostdate_{lead.id}")

                    # Save button
                    save_btn = st.form_submit_button("Save Lead")
                    if save_btn:
                        try:
                            # Apply changes to SQLAlchemy object
                            lead.contact_name = contact_name.strip() or None
                            lead.contact_phone = contact_phone.strip() or None
                            lead.contact_email = contact_email.strip() or None
                            lead.property_address = property_address.strip() or None
                            lead.damage_type = damage_type
                            lead.assigned_to = assigned_to.strip() or None
                            lead.estimated_value = float(est_val) if est_val else None
                            lead.notes = notes.strip() or None
                            lead.sla_hours = int(sla_hours)
                            # pipeline flags
                            lead.contacted = True if contacted_choice == "Yes" else False
                            lead.inspection_scheduled = True if inspection_scheduled_choice == "Yes" else False
                            lead.inspection_scheduled_at = inspection_dt
                            lead.inspection_completed = True if inspection_completed_choice == "Yes" else False
                            lead.inspection_completed_at = inspection_comp_dt
                            lead.estimate_submitted = True if estimate_sub_choice == "Yes" else False
                            lead.estimate_submitted_at = (datetime.combine(est_submitted_at, datetime.min.time()) if est_submitted_at else None)
                            lead.awarded_comment = awarded_comment.strip() or None
                            lead.awarded_date = (datetime.combine(awarded_date, datetime.min.time()) if awarded_comment or awarded_date else None)
                            lead.lost_comment = lost_comment.strip() or None
                            lead.lost_date = (datetime.combine(lost_date, datetime.min.time()) if lost_comment or lost_date else None)
                            # status change
                            if status_choice != lead.status:
                                lead.status = status_choice
                                lead.sla_stage = status_choice
                                lead.sla_entered_at = datetime.utcnow()
                            s.add(lead)
                            s.commit()
                            st.success(f"Lead #{lead.id} saved.")
                        except Exception as e:
                            st.error(f"Error saving lead #{lead.id}: {e}")

                # Estimates section (outside the edit form to avoid nested form issues)
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
                    # actions for first estimate
                    first_est = ests[0]
                    ea, eb, ec = st.columns(3)
                    with ea:
                        if st.button(f"Mark Sent (#{first_est.id})", key=f"send_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_sent(s, first_est.id)
                                st.success("Marked as sent.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with eb:
                        if st.button(f"Mark Approved (#{first_est.id})", key=f"app_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_approved(s, first_est.id)
                                st.success("Marked approved; lead moved to Awarded.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with ec:
                        if st.button(f"Mark Lost (#{first_est.id})", key=f"lost_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_lost(s, first_est.id, reason="Lost to competitor")
                                st.success("Marked lost; lead moved to Lost.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                else:
                    st.write("No estimates yet.")
                    # create estimate form
                    with st.form(f"create_est_{lead.id}", clear_on_submit=True):
                        amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"new_est_amt_{lead.id}")
                        det = st.text_area("Estimate details (optional)", key=f"new_est_det_{lead.id}")
                        create_btn = st.form_submit_button("Create Estimate")
                        if create_btn:
                            try:
                                create_estimate(s, lead.id, float(amt), details=det)
                                st.success("Estimate created.")
                            except Exception as e:
                                st.error(f"Failed to create estimate: {e}")

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
        fig = px.bar(funnel, x="stage", y="count", title="Leads by Stage", text="count")
        fig.update_layout(xaxis_title=None, yaxis_title="Number of Leads", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Summary note
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

        # Conversion by source
        st.subheader("Conversion by Source")
        conv = df.copy()
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(leads=("id", "count"), awarded=("awarded_flag", "sum")).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        # SLA overdue
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

# --- Page: Exports
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode('utf-8')
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
        st.markdown("---")
        st.subheader("Estimates")
        df_est = estimates_df(s)
        if not df_est.empty:
            st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode('utf-8'), file_name="estimates.csv", mime="text/csv")

# End of file
