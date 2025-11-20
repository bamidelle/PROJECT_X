# project_x_migration_app_v3.py
"""
Project X — Migration Ready v3
- Inputs: user-entered text shown in deep black
- Priority weight tuning UI in sidebar (adjust weights in real time)
- Priority Leads summary in Pipeline Board (red/orange/white)
- Quick contact actions (Call, WhatsApp, Email) for each lead
- Generates an Alembic-ready migration script (downloadable)
- Keeps safe in-app ALTER migration as fallback
"""

import os
from datetime import datetime, timedelta
import io

import streamlit as st
import pandas as pd
import plotly.express as px

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# CONFIG
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_migration_v3.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# MIGRATION COLUMNS (for in-app ALTER and Alembic script generator)
# ---------------------------
MIGRATION_COLUMNS = {
    "contacted": ("BOOLEAN", "0"),
    "inspection_scheduled": ("BOOLEAN", "0"),
    "inspection_scheduled_at": ("DATETIME", None),
    "inspection_completed": ("BOOLEAN", "0"),
    "inspection_completed_at": ("DATETIME", None),
    "estimate_submitted": ("BOOLEAN", "0"),
    "estimate_submitted_at": ("DATETIME", None),
    "awarded_comment": ("TEXT", None),
    "awarded_date": ("DATE", None),
    "lost_comment": ("TEXT", None),
    "lost_date": ("DATE", None),
}

# ---------------------------
# UI CSS: Roboto font, white labels, deep-black user input text, dark placeholders
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
  font-size: 20px;
}

/* panels */
.panel {
  background: rgba(255,255,255,0.03);
  padding: 12px;
  border-radius: var(--radius);
  border: 1px solid rgba(255,255,255,0.04);
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
# ORM MODELS (with pipeline fields)
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

    # source / contact
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

    # pipeline fields (new)
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
    history = relationship("StageHistory", back_populates="lead", cascade="all, delete-orphan")


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


class StageHistory(Base):
    __tablename__ = "stage_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    from_stage = Column(String)
    to_stage = Column(String)
    changed_by = Column(String, nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow)

    lead = relationship("Lead", back_populates="history")

# ---------------------------
# DB init + in-app safe migration helpers
# ---------------------------
def create_tables_if_missing():
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "leads" not in inspector.get_table_names():
        return
    existing_cols = {col["name"] for col in inspector.get_columns("leads")}
    conn = engine.connect()
    for col_name, (col_type, default) in MIGRATION_COLUMNS.items():
        if col_name not in existing_cols:
            # Choose SQL based on type
            if col_type == "BOOLEAN":
                col_def = "INTEGER DEFAULT 0"
            elif col_type == "DATETIME":
                col_def = "TEXT"
            elif col_type == "DATE":
                col_def = "TEXT"
            else:
                col_def = "TEXT"
            sql = f'ALTER TABLE leads ADD COLUMN {col_name} {col_def};'
            try:
                conn.execute(sql)
            except Exception as e:
                print(f"Migration add column {col_name} failed: {e}")
    conn.close()

def init_db():
    create_tables_if_missing()

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
    hist = StageHistory(lead_id=lead.id, from_stage=None, to_stage=LeadStatus.NEW, changed_by=kwargs.get("created_by", "system"))
    session.add(hist)
    session.commit()
    return lead

def change_stage(session, lead_id, new_stage, changed_by="user"):
    lead = session.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        return None
    old = lead.status
    lead.status = new_stage
    lead.sla_stage = new_stage
    lead.sla_entered_at = datetime.utcnow()
    session.add(lead)
    hist = StageHistory(lead_id=lead.id, from_stage=old, to_stage=new_stage, changed_by=changed_by)
    session.add(hist)
    session.commit()
    return lead

def create_estimate(session, lead_id, amount, details=""):
    est = Estimate(lead_id=lead_id, amount=amount, created_at=datetime.utcnow(), details=details)
    session.add(est)
    session.commit()
    return est

def mark_estimate_sent(session, estimate_id):
    est = session.query(Estimate).filter(Estimate.id == estimate_id).first()
    est.sent_at = datetime.utcnow()
    session.add(est)
    session.commit()
    return est

def mark_estimate_approved(session, estimate_id):
    est = session.query(Estimate).filter(Estimate.id == estimate_id).first()
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
    est.lost = True
    est.lost_reason = reason
    session.add(est)
    lead = est.lead
    lead.status = LeadStatus.LOST
    lead.lost_date = datetime.utcnow()
    session.add(lead)
    session.commit()
    return est

def leads_df(session):
    return pd.read_sql(session.query(Lead).statement, session.bind)

def estimates_df(session):
    return pd.read_sql(session.query(Estimate).statement, session.bind)

# ---------------------------
# Priority scoring (uses weights from sidebar)
# ---------------------------
def compute_priority_for_lead_row(lead_row, weights):
    """
    weights: dict with keys:
      - value_weight (0..1)
      - sla_weight (0..1)
      - urgency_weight (0..1)
      - contacted_weight (0..1)
      - inspection_weight (0..1)
      - estimate_weight (0..1)
    Returns score 0..1 and components.
    """
    # estimated value normalization: baseline 5000
    val = float(lead_row.get("estimated_value") or 0.0)
    value_score = min(val / (weights.get("value_baseline", 5000.0)), 1.0)

    # time left calculation (hours)
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

    # sla urgency score: 0..1 where <=0 =>1, 72 hours =>0
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)

    # urgency flags
    contacted = bool(lead_row.get("contacted"))
    inspection_scheduled = bool(lead_row.get("inspection_scheduled"))
    estimate_submitted = bool(lead_row.get("estimate_submitted"))

    # flags scoring (1 if missing, 0 if done)
    contacted_flag = 0.0 if contacted else 1.0
    inspection_flag = 0.0 if inspection_scheduled else 1.0
    estimate_flag = 0.0 if estimate_submitted else 1.0

    # weighted sum (normalize by total weight)
    total_weight = (weights["value_weight"] + weights["sla_weight"] + weights["urgency_weight"])
    if total_weight <= 0:
        total_weight = 1.0
    score = (
        value_score * weights["value_weight"]
        + sla_score * weights["sla_weight"]
        + (contacted_flag * weights["contacted_w"] + inspection_flag * weights["inspection_w"] + estimate_flag * weights["estimate_w"]) * weights["urgency_weight"]
    ) / total_weight

    # clamp 0..1
    score = max(0.0, min(score, 1.0))
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

# ---------------------------
# Alembic migration generator helper
# ---------------------------
def generate_alembic_migration_script():
    """
    Returns a string containing an Alembic-style migration script that adds
    the columns in MIGRATION_COLUMNS to the 'leads' table.
    Paste into an Alembic revision script (inside upgrade()) and run alembic upgrade head.
    """
    imports = (
        "from alembic import op\n"
        "import sqlalchemy as sa\n\n"
    )
    header = (
        "\"\"\"auto migration: add pipeline fields\n\nRevision ID: add_pipeline_fields\nRevises: \nCreate Date: " + datetime.utcnow().isoformat() + "\n\"\"\"\n\n"
        "revision = 'add_pipeline_fields'\n"
        "down_revision = None\n"
        "branch_labels = None\n"
        "depends_on = None\n\n"
    )
    upgrade_lines = []
    for col, (typ, default) in MIGRATION_COLUMNS.items():
        # Choose sa type
        if typ == "BOOLEAN":
            sa_type = "sa.Boolean()"
        elif typ == "DATETIME":
            sa_type = "sa.DateTime()"
        elif typ == "DATE":
            sa_type = "sa.Date()"
        else:
            sa_type = "sa.Text()"
        upgrade_lines.append(f"    op.add_column('leads', sa.Column('{col}', {sa_type}, nullable=True))")
    upgrade_block = "def upgrade():\n" + "\n".join(upgrade_lines) + "\n\n"
    downgrade_lines = []
    for col in MIGRATION_COLUMNS.keys():
        downgrade_lines.append(f"    op.drop_column('leads', '{col}')")
    downgrade_block = "def downgrade():\n" + "\n".join(downgrade_lines) + "\n"
    return imports + header + upgrade_block + downgrade_block

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Project X — Migration Ready v3", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# Init DB (create + migrate lightly)
init_db()

# Header
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker (v3)</div>", unsafe_allow_html=True)

# Sidebar: controls + priority weight tuning + migration generator
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=1)
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
            estimated_value=4200.0,
            notes="Demo lead",
            sla_hours=24,
            created_by="demo"
        )
        st.success("Demo lead added")
    st.markdown("---")
    st.markdown("### Priority Weight Tuning")
    # Persist weights in session_state to allow reuse
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5,
            "sla_weight": 0.35,
            "urgency_weight": 0.15,
            "contacted_w": 0.6,
            "inspection_w": 0.5,
            "estimate_w": 0.5,
            "value_baseline": 5000.0
        }
    w = st.session_state.weights
    w["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(w["value_weight"]), step=0.05)
    w["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(w["sla_weight"]), step=0.05)
    w["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(w["urgency_weight"]), step=0.05)
    st.markdown("**Within urgency flags:**")
    w["contacted_w"] = st.slider("  Not-contacted weight", 0.0, 1.0, float(w["contacted_w"]), step=0.05)
    w["inspection_w"] = st.slider("  Not-scheduled weight", 0.0, 1.0, float(w["inspection_w"]), step=0.05)
    w["estimate_w"] = st.slider("  No-estimate weight", 0.0, 1.0, float(w["estimate_w"]), step=0.05)
    w["value_baseline"] = st.number_input("Value baseline (for normalization)", min_value=100.0, value=float(w["value_baseline"]), step=100.0)

    st.markdown("---")
    st.markdown("### Migration")
    st.markdown("Generate an Alembic-style migration script to add pipeline fields (optional).")
    if st.button("Generate Alembic migration script"):
        script = generate_alembic_migration_script()
        st.download_button("Download Alembic script", data=script.encode("utf-8"), file_name="alembic_add_pipeline_fields.py", mime="text/x-python")
        st.success("Alembic script generated (download below).")
    st.markdown("---")
    st.markdown(f"DB: <small>{DB_FILE}</small>", unsafe_allow_html=True)

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
                sla_hours=int(sla_hours),
                created_by="web_user"
            )
            st.success(f"Lead created (ID: {lead.id})")
            st.rerun()

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values('created_at', ascending=False).head(50))

# --- Page: Pipeline Board
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet.")
    else:
        # compute priorities using current weights
        df = leads_df(s)
        weights = st.session_state.weights
        priority_list = []
        for _, row in df.iterrows():
            score, value_score, sla_score, contacted_flag, inspection_flag, estimation_flag, time_left_h = compute_priority_for_lead_row(row, weights)
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
        st.subheader("Priority Leads")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                if score >= 0.7:
                    color = "red"
                elif score >= 0.45:
                    color = "orange"
                else:
                    color = "white"
                st.markdown(
                    f\"\"\"<div style='padding:8px;border-radius:8px;margin-bottom:6px;border:1px solid rgba(255,255,255,0.04);'>
                    <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                    <span style='color:var(--muted);'> | Est: ${r['estimated_value']:,.0f}</span>
                    <span style='color:var(--muted);'> | Time left: {int(r['time_left_hours'])}h</span>
                    <span style='float:right;color:{color};'>Priority: {r['priority_score']:.2f}</span>
                    </div>\"\"\", unsafe_allow_html=True
                )
        else:
            st.info("No priority leads yet.")
        st.markdown("---")

        # Render each lead as the original stage-buckets style but with quick-contact and ability to edit in expander
        cols = st.columns(len(LeadStatus.ALL))
        buckets = {stg: [] for stg in LeadStatus.ALL}
        for l in leads:
            buckets[l.status].append(l)

        for i, stg in enumerate(LeadStatus.ALL):
            with cols[i]:
                st.markdown(f"### {stg} ({len(buckets[stg])})")
                for lead in buckets[stg]:
                    with st.expander(f"#{lead.id} · {lead.contact_name or 'No name'} · ${lead.estimated_value or 0:,.0f}"):
                        st.write(f"**Source:** {lead.source} · **Assigned:** {lead.assigned_to}")
                        st.write(f"**Address:** {lead.property_address}")
                        st.write(f"**Damage:** {lead.damage_type}")
                        st.write(f"**Notes:** {lead.notes}")
                        st.write(f"**Created:** {lead.created_at}")

                        # quick contact actions (show only when data available)
                        qc_cols = st.columns(3)
                        phone = (lead.contact_phone or "").strip()
                        email = (lead.contact_email or "").strip()
                        if phone:
                            tel_link = f"tel:{phone}"
                            wa_link = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                            qc_cols[0].markdown(f\"<a href='{tel_link}'><button>📞 Call</button></a>\", unsafe_allow_html=True)
                            qc_cols[1].markdown(f\"<a href='{wa_link}' target='_blank'><button>💬 WhatsApp</button></a>\", unsafe_allow_html=True)
                        else:
                            qc_cols[0].write(\" \")
                            qc_cols[1].write(\" \")
                        if email:
                            mail_link = f\"mailto:{email}?subject=Follow%20up%20on%20your%20restoration%20request\"
                            qc_cols[2].markdown(f\"<a href='{mail_link}'><button>✉️ Email</button></a>\", unsafe_allow_html=True)
                        else:
                            qc_cols[2].write(\" \")

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
                            st.markdown(f\"❗ **SLA OVERDUE** (was due {deadline.strftime('%Y-%m-%d %H:%M')})\")
                        else:
                            st.markdown(f\"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})\")

                        st.markdown(\"---\")
                        # Editable pipeline fields inside expander
                        with st.form(f\"lead_edit_{lead.id}\"):
                            contact_name = st.text_input(\"Contact name\", value=lead.contact_name or \"\", key=f\"cname_{lead.id}\")
                            contact_phone = st.text_input(\"Contact phone\", value=lead.contact_phone or \"\", key=f\"cphone_{lead.id}\")
                            contact_email = st.text_input(\"Contact email\", value=lead.contact_email or \"\", key=f\"cemail_{lead.id}\")
                            assigned_to = st.text_input(\"Assigned to\", value=lead.assigned_to or \"\", key=f\"assign_{lead.id}\")
                            est_val = st.number_input(\"Estimated value (USD)\", min_value=0.0, value=float(lead.estimated_value or 0.0), step=50.0, key=f\"est_{lead.id}\")
                            notes = st.text_area(\"Notes\", value=lead.notes or \"\", key=f\"notes_{lead.id}\")
                            contacted_choice = st.selectbox(\"Contacted?\", [\"No\",\"Yes\"], index=1 if lead.contacted else 0, key=f\"cont_{lead.id}\")
                            insp_sch_choice = st.selectbox(\"Inspection Scheduled?\", [\"No\",\"Yes\"], index=1 if lead.inspection_scheduled else 0, key=f\"inspsch_{lead.id}\")
                            if insp_sch_choice == \"Yes\":
                                default_dt = lead.inspection_scheduled_at or datetime.utcnow()
                                try:
                                    if isinstance(default_dt, str):
                                        default_dt = datetime.fromisoformat(default_dt)
                                except Exception:
                                    default_dt = datetime.utcnow()
                                inspection_dt = st.datetime_input(\"Inspection date & time\", value=default_dt, key=f\"insp_dt_{lead.id}\")
                            else:
                                inspection_dt = None
                            insp_comp_choice = st.selectbox(\"Inspection Completed?\", [\"No\",\"Yes\"], index=1 if lead.inspection_completed else 0, key=f\"inspcomp_{lead.id}\")
                            if insp_comp_choice == \"Yes\":
                                default_dt2 = lead.inspection_completed_at or datetime.utcnow()
                                try:
                                    if isinstance(default_dt2, str):
                                        default_dt2 = datetime.fromisoformat(default_dt2)
                                except Exception:
                                    default_dt2 = datetime.utcnow()
                                inspection_comp_dt = st.datetime_input(\"Inspection completed at\", value=default_dt2, key=f\"insp_comp_dt_{lead.id}\")
                            else:
                                inspection_comp_dt = None
                            est_sub_choice = st.selectbox(\"Estimate Submitted?\", [\"No\",\"Yes\"], index=1 if lead.estimate_submitted else 0, key=f\"estsub_{lead.id}\")
                            awarded_comment = st.text_input(\"Awarded comment (optional)\", value=lead.awarded_comment or \"\", key=f\"awcom_{lead.id}\")
                            awarded_date = st.date_input(\"Awarded date (optional)\", value=(lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()), key=f\"awdate_{lead.id}\")
                            lost_comment = st.text_input(\"Lost comment (optional)\", value=lead.lost_comment or \"\", key=f\"lostcom_{lead.id}\")
                            lost_date = st.date_input(\"Lost date (optional)\", value=(lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()), key=f\"lostdate_{lead.id}\")
                            new_stage = st.selectbox(\"Move lead stage\", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f\"stage_{lead.id}\")
                            save = st.form_submit_button(\"Save Lead\")
                            if save:
                                try:
                                    lead.contact_name = contact_name.strip() or None
                                    lead.contact_phone = contact_phone.strip() or None
                                    lead.contact_email = contact_email.strip() or None
                                    lead.assigned_to = assigned_to.strip() or None
                                    lead.estimated_value = float(est_val) if est_val else None
                                    lead.notes = notes.strip() or None
                                    lead.contacted = True if contacted_choice == \"Yes\" else False
                                    lead.inspection_scheduled = True if insp_sch_choice == \"Yes\" else False
                                    lead.inspection_scheduled_at = inspection_dt
                                    lead.inspection_completed = True if insp_comp_choice == \"Yes\" else False
                                    lead.inspection_completed_at = inspection_comp_dt
                                    lead.estimate_submitted = True if est_sub_choice == \"Yes\" else False
                                    lead.awarded_comment = awarded_comment.strip() or None
                                    lead.awarded_date = datetime.combine(awarded_date, datetime.min.time()) if awarded_comment or awarded_date else None
                                    lead.lost_comment = lost_comment.strip() or None
                                    lead.lost_date = datetime.combine(lost_date, datetime.min.time()) if lost_comment or lost_date else None
                                    if new_stage != lead.status:
                                        lead.status = new_stage
                                        lead.sla_stage = new_stage
                                        lead.sla_entered_at = datetime.utcnow()
                                    s.add(lead)
                                    s.commit()
                                    st.success(f\"Lead #{lead.id} updated\")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f\"Failed saving lead: {e}\")

                        # Estimates and estimate actions below
                        st.markdown(\"---\")
                        st.markdown(\"Estimates\")
                        ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).all()
                        if ests:
                            est_df = pd.DataFrame([{
                                \"id\": e.id, \"amount\": e.amount, \"sent_at\": e.sent_at, \"approved\": e.approved, \"lost\": e.lost, \"lost_reason\": e.lost_reason
                            } for e in ests])
                            st.dataframe(est_df)
                        else:
                            st.write(\"No estimates yet.\")
                        with st.form(f\"est_form_{lead.id}\", clear_on_submit=True):
                            amt = st.number_input(\"Estimate amount (USD)\", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f\"est_amt_{lead.id}\")
                            details = st.text_area(\"Estimate details\", key=f\"est_det_{lead.id}\")
                            if st.form_submit_button(\"Create Estimate\", key=f\"est_submit_{lead.id}\"):
                                create_estimate(s, lead.id, float(amt), details=details)
                                st.success(\"Estimate created\")
                                st.rerun()

                        if ests:
                            first_est = ests[0]
                            ca, cb, cc = st.columns(3)
                            with ca:
                                if st.button(\"Mark Sent\", key=f\"send_{first_est.id}\"):
                                    mark_estimate_sent(s, first_est.id)
                                    st.success(\"Marked as sent\")
                                    st.rerun()
                            with cb:
                                if st.button(\"Mark Approved\", key=f\"app_{first_est.id}\"):
                                    mark_estimate_approved(s, first_est.id)
                                    st.success(\"Estimate approved and lead marked Awarded\")
                                    st.rerun()
                            with cc:
                                if st.button(\"Mark Lost\", key=f\"lost_{first_est.id}\"):
                                    mark_estimate_lost(s, first_est.id, reason=\"Lost to competitor\")
                                    st.success(\"Estimate marked lost and lead moved to Lost\")
                                    st.rerun()

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

        # Summary & top priority (use same weighting)
        st.markdown("### Summary & Priority")
        total_leads = len(df)
        awarded = len(df[df.status == LeadStatus.AWARDED])
        lost = len(df[df.status == LeadStatus.LOST])
        contacted_cnt = len(df[df.contacted == True]) if "contacted" in df.columns else 0
        insp_sched_cnt = len(df[df.inspection_scheduled == True]) if "inspection_scheduled" in df.columns else 0
        st.markdown(f"- Total leads: **{total_leads}**")
        st.markdown(f"- Awarded: **{awarded}**")
        st.markdown(f"- Lost: **{lost}**")
        st.markdown(f"- Contacted: **{contacted_cnt}**")
        st.markdown(f"- Inspections scheduled: **{insp_sched_cnt}**")

        # compute and list top priority leads
        weights = st.session_state.weights
        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left),
                "priority_score": score,
                "status": row.get("status")
            })
        p_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)
        st.subheader("Top Priority Leads (by score)")
        if not p_df.empty:
            for _, r in p_df.head(10).iterrows():
                color = "red" if r["priority_score"] >= 0.7 else ("orange" if r["priority_score"] >= 0.45 else "white")
                st.markdown(
                    f\"\"\"<div style='padding:8px;border-radius:8px;margin-bottom:6px;border:1px solid rgba(255,255,255,0.04);'>
                    <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                    <span style='color:var(--muted);'> | Est: ${r['estimated_value']:,.0f}</span>
                    <span style='color:var(--muted);'> | Time left: {int(r['time_left_hours'])}h</span>
                    <span style='float:right;color:{color};'>Priority: {r['priority_score']:.2f}</span>
                    </div>\"\"\", unsafe_allow_html=True
                )
        else:
            st.info("No priority leads.")

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
    df_hist = pd.read_sql(s.query(StageHistory).statement, s.bind)
    df_est = estimates_df(s)
    if not df_leads.empty:
        csv = df_leads.to_csv(index=False).encode('utf-8')
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    else:
        st.info("No leads yet to export.")
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode('utf-8'), file_name="estimates.csv", mime="text/csv")
    if not df_hist.empty:
        st.download_button("Download stage_history.csv", df_hist.to_csv(index=False).encode('utf-8'), file_name="stage_history.csv", mime="text/csv")

# End of file
