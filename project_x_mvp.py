# project_x_migration_app.py
"""
Project X — Expanded with Safe Migration and Updated Pipeline Fields

Features:
- SQLAlchemy ORM models with new pipeline fields
- Safe in-place migration for older DBs (adds missing columns)
- Pipeline Board inputs:
  - Contacted (Yes/No)
  - Inspection Scheduled (Yes/No) + date/time
  - Inspection Completed (Yes/No) + date/time
  - Estimate Submitted (Yes/No) optional + date/time
  - Awarded (comment + date)
  - Lost (comment + date)
- Funnel Overview reflects pipeline data + summary notes
- Google Font (Roboto), white text, dark grey placeholders
- Buttons: transparent, white border, white text
- SLA calculation fixed (uses datetime.utcnow())
- Streamlit friendly (uses st.rerun())

Run:
pip install streamlit sqlalchemy pandas plotly
streamlit run project_x_migration_app.py
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
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_migration.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# SAFE MIGRATION: list of new columns to ensure exist (name -> SQL)
# SQLite type choices used by ALTER TABLE ADD COLUMN: INTEGER, REAL, TEXT
MIGRATION_COLUMNS = {
    # boolean fields -> INTEGER (0/1)
    "contacted": "INTEGER DEFAULT 0",
    "inspection_scheduled": "INTEGER DEFAULT 0",
    "inspection_scheduled_at": "TEXT",  # ISO datetime string
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
# GOOGLE-STYLE UI CSS (Roboto), white text, dark placeholders, transparent buttons
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
  font-weight: 500;
  font-size: 20px;
}

/* panels */
.panel {
  background: rgba(255,255,255,0.03);
  padding: 12px;
  border-radius: var(--radius);
  border: 1px solid rgba(255,255,255,0.04);
}

/* all text white */
div, p, span, label {
  color: var(--white) !important;
}

/* placeholder color */
input::placeholder, textarea::placeholder {
  color: var(--placeholder) !important;
}

/* form inputs */
input, textarea, select {
  background: rgba(255,255,255,0.01) !important;
  color: var(--white) !important;
  border-radius: 8px !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
}

/* transparent button with white border */
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
# ORM MODELS (with new pipeline fields)
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

    # NEW pipeline fields (may be absent in older DB; migration must add them)
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
# DB init + safe migration helpers
# ---------------------------
def create_tables_if_missing():
    """
    Create tables (if they don't exist) and then perform safe ALTER TABLE
    to add any missing columns listed in MIGRATION_COLUMNS.
    This function is idempotent and preserves existing data.
    """
    # Create tables if they don't exist according to ORM
    Base.metadata.create_all(bind=engine)

    # Use pragma to inspect existing columns in leads table
    inspector = inspect(engine)
    if "leads" not in inspector.get_table_names():
        # Nothing to migrate beyond creating tables
        return

    existing_cols = {col["name"] for col in inspector.get_columns("leads")}
    # Add missing columns with raw SQL (SQLite supports ALTER TABLE ADD COLUMN <col_def>)
    conn = engine.connect()
    for col_name, col_def in MIGRATION_COLUMNS.items():
        if col_name not in existing_cols:
            sql = f'ALTER TABLE leads ADD COLUMN {col_name} {col_def};'
            try:
                conn.execute(sql)
                # SQLite doesn't update SQLAlchemy metadata automatically; that's okay
            except Exception as e:
                # Log and continue — don't fail migration for a single column
                print(f"Migration add column {col_name} failed: {e}")
    conn.close()


def init_db():
    """
    Initialize DB and perform migration.
    """
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
        # pipeline defaults (if provided)
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
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Project X — Migration Ready", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# Initialize DB and migrate if needed
init_db()

# Header
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(
            s,
            source="Google Ads",
            source_details="gclid=demo",
            contact_name="Demo Customer",
            contact_phone="+1-555-0000",
            contact_email="demo@example.com",
            property_address="100 Demo Ave",
            damage_type="water",
            assigned_to="Alex",
            estimated_value=1200.0,
            notes="Demo lead",
            sla_hours=12,
            created_by="demo"
        )
        st.success("Demo lead added")
    st.markdown("---")
    st.markdown(f"DB file: <small>{DB_FILE}</small>", unsafe_allow_html=True)
    st.markdown("Tips:")
    st.markdown("- Use Pipeline Board to update pipeline fields.")
    st.markdown("- Funnel Overview reflects live pipeline data.")


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

                        # SLA calculation (use datetime.utcnow())
                        entered = lead.sla_entered_at or lead.created_at
                        if isinstance(entered, str):
                            try:
                                entered = datetime.fromisoformat(entered)
                            except Exception:
                                entered = datetime.utcnow()
                        deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                        remaining = deadline - datetime.utcnow()
                        if remaining.total_seconds() <= 0:
                            st.markdown(f"❗ **SLA OVERDUE** (was due {deadline.strftime('%Y-%m-%d %H:%M')})")
                        else:
                            st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                        st.markdown("---")
                        # Contacted (Yes/No)
                        contacted_choice = st.selectbox("Contacted?", ["No", "Yes"], index=1 if lead.contacted else 0, key=f"contacted_{lead.id}")
                        new_contacted = True if contacted_choice == "Yes" else False
                        if new_contacted != bool(lead.contacted):
                            lead.contacted = new_contacted
                            s.add(lead); s.commit()
                            st.success("Contacted status updated")

                        # Inspection scheduled (Yes/No) + datetime
                        insp_sch_choice = st.selectbox("Inspection Scheduled?", ["No", "Yes"], index=1 if lead.inspection_scheduled else 0, key=f"insp_sch_{lead.id}")
                        insp_scheduled = True if insp_sch_choice == "Yes" else False
                        insp_dt = None
                        if insp_scheduled:
                            # default to stored datetime or now
                            default_dt = lead.inspection_scheduled_at or datetime.utcnow()
                            try:
                                # If stored as string, parse
                                if isinstance(default_dt, str):
                                    default_dt = datetime.fromisoformat(default_dt)
                            except Exception:
                                default_dt = datetime.utcnow()
                            insp_dt = st.datetime_input("Inspection date & time", value=default_dt, key=f"insp_dt_{lead.id}")
                        # Save inspection scheduled changes
                        if (insp_scheduled != bool(lead.inspection_scheduled)) or (insp_dt and lead.inspection_scheduled_at != insp_dt):
                            lead.inspection_scheduled = insp_scheduled
                            lead.inspection_scheduled_at = insp_dt
                            s.add(lead); s.commit()
                            st.success("Inspection schedule updated")

                        # Inspection completed (Yes/No) + datetime
                        insp_comp_choice = st.selectbox("Inspection Completed?", ["No", "Yes"], index=1 if lead.inspection_completed else 0, key=f"insp_comp_{lead.id}")
                        insp_comp = True if insp_comp_choice == "Yes" else False
                        insp_comp_dt = None
                        if insp_comp:
                            default_dt = lead.inspection_completed_at or datetime.utcnow()
                            try:
                                if isinstance(default_dt, str):
                                    default_dt = datetime.fromisoformat(default_dt)
                            except Exception:
                                default_dt = datetime.utcnow()
                            insp_comp_dt = st.datetime_input("Inspection completed at", value=default_dt, key=f"insp_comp_dt_{lead.id}")
                        if (insp_comp != bool(lead.inspection_completed)) or (insp_comp_dt and lead.inspection_completed_at != insp_comp_dt):
                            lead.inspection_completed = insp_comp
                            lead.inspection_completed_at = insp_comp_dt
                            s.add(lead); s.commit()
                            st.success("Inspection completed status updated")

                        # Estimate submitted (Yes/No) optional + datetime
                        est_sub_choice = st.selectbox("Estimate Submitted?", ["No", "Yes"], index=1 if lead.estimate_submitted else 0, key=f"est_sub_{lead.id}")
                        est_sub = True if est_sub_choice == "Yes" else False
                        est_sub_dt = None
                        if est_sub:
                            default_dt = lead.estimate_submitted_at or datetime.utcnow()
                            try:
                                if isinstance(default_dt, str):
                                    default_dt = datetime.fromisoformat(default_dt)
                            except Exception:
                                default_dt = datetime.utcnow()
                            est_sub_dt = st.datetime_input("Estimate submitted at (optional)", value=default_dt, key=f"est_sub_dt_{lead.id}")
                        if (est_sub != bool(lead.estimate_submitted)) or (est_sub_dt and lead.estimate_submitted_at != est_sub_dt):
                            lead.estimate_submitted = est_sub
                            lead.estimate_submitted_at = est_sub_dt
                            s.add(lead); s.commit()
                            st.success("Estimate submission status updated")

                        # Awarded: comment + date
                        st.markdown("Awarded (optional)")
                        awarded_comment = st.text_input("Awarded comment (optional)", value=lead.awarded_comment or "", key=f"award_com_{lead.id}")
                        # awarded_date uses date input (date only)
                        awarded_date_val = lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()
                        awarded_date = st.date_input("Awarded date (optional)", value=awarded_date_val, key=f"award_date_{lead.id}")
                        if (awarded_comment != (lead.awarded_comment or "")) or (lead.awarded_date is None or lead.awarded_date.date() != awarded_date):
                            lead.awarded_comment = awarded_comment if awarded_comment.strip() else None
                            lead.awarded_date = datetime.combine(awarded_date, datetime.min.time())
                            if awarded_comment or lead.awarded_date:
                                lead.status = LeadStatus.AWARDED
                            s.add(lead); s.commit()
                            st.success("Awarded info updated")

                        # Lost: comment + date
                        st.markdown("Lost (optional)")
                        lost_comment = st.text_input("Lost comment (optional)", value=lead.lost_comment or "", key=f"lost_com_{lead.id}")
                        lost_date_val = lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()
                        lost_date = st.date_input("Lost date (optional)", value=lost_date_val, key=f"lost_date_{lead.id}")
                        if (lost_comment != (lead.lost_comment or "")) or (lead.lost_date is None or lead.lost_date.date() != lost_date):
                            lead.lost_comment = lost_comment if lost_comment.strip() else None
                            lead.lost_date = datetime.combine(lost_date, datetime.min.time())
                            if lost_comment or lead.lost_date:
                                lead.status = LeadStatus.LOST
                            s.add(lead); s.commit()
                            st.success("Lost info updated")

                        st.markdown("---")
                        # Stage change UI
                        new_stage = st.selectbox("Move lead stage", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"stage_{lead.id}")
                        if st.button("Change Stage", key=f"btn_change_{lead.id}"):
                            change_stage(s, lead.id, new_stage, changed_by="ui_user")
                            st.success(f"Lead #{lead.id} moved to {new_stage}")
                            st.rerun()

                        # Estimates (existing)
                        st.markdown("Estimates")
                        ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).all()
                        if ests:
                            est_df = pd.DataFrame([{
                                "id": e.id, "amount": e.amount, "sent_at": e.sent_at, "approved": e.approved, "lost": e.lost, "lost_reason": e.lost_reason
                            } for e in ests])
                            st.dataframe(est_df)
                        else:
                            st.write("No estimates yet.")

                        with st.form(f"est_form_{lead.id}", clear_on_submit=True):
                            amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"est_amt_{lead.id}")
                            details = st.text_area("Estimate details", key=f"est_det_{lead.id}")
                            if st.form_submit_button("Create Estimate", key=f"est_submit_{lead.id}"):
                                create_estimate(s, lead.id, float(amt), details=details)
                                st.success("Estimate created")
                                st.rerun()

                        # Buttons for first estimate
                        if ests:
                            first_est = ests[0]
                            ca, cb, cc = st.columns(3)
                            with ca:
                                if st.button("Mark Sent", key=f"send_{first_est.id}"):
                                    mark_estimate_sent(s, first_est.id)
                                    st.success("Marked as sent")
                                    st.rerun()
                            with cb:
                                if st.button("Mark Approved", key=f"app_{first_est.id}"):
                                    mark_estimate_approved(s, first_est.id)
                                    st.success("Estimate approved and lead marked Awarded")
                                    st.rerun()
                            with cc:
                                if st.button("Mark Lost", key=f"lost_{first_est.id}"):
                                    mark_estimate_lost(s, first_est.id, reason="Lost to competitor")
                                    st.success("Estimate marked lost and lead moved to Lost")
                                    st.rerun()


# --- Page: Analytics & SLA
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Funnel: counts by status (reflect pipeline board)
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
        contacted_cnt = len(df[df.contacted == True]) if "contacted" in df.columns else 0
        insp_sched_cnt = len(df[df.inspection_scheduled == True]) if "inspection_scheduled" in df.columns else 0
        st.markdown(f"- Total leads: **{total_leads}**")
        st.markdown(f"- Awarded: **{awarded}**")
        st.markdown(f"- Lost: **{lost}**")
        st.markdown(f"- Contacted: **{contacted_cnt}**")
        st.markdown(f"- Inspections scheduled: **{insp_sched_cnt}**")

        # Top lost comments sample
        lost_recs = df[df.status == LeadStatus.LOST]
        if not lost_recs.empty and "lost_comment" in lost_recs.columns:
            top_lost = lost_recs[~lost_recs.lost_comment.isna()].lost_comment.value_counts().head(5)
            if not top_lost.empty:
                st.markdown("- Top lost comments (sample):")
                for idx, cnt in top_lost.items():
                    st.markdown(f"  - {idx} ({cnt})")
        else:
            st.markdown("- No lost leads with comments yet.")

        # Conversion by source
        st.subheader("Conversion by Source")
        conv = df.copy()
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(leads=("id", "count"), awarded=("awarded_flag", "sum")).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        # SLA overview: list overdue leads
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
