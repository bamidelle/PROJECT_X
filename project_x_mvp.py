# project_x_expanded.py
"""
Project X - Expanded MVP (SQLAlchemy ORM)
Single-file Streamlit app implementing:
- Lead intake (SQLAlchemy models)
- Pipeline board & stage transitions
- Funnel analytics & conversion by source
- SLA countdowns and overdue highlighting
- CSV exports 
- Clean Grok-inspired UI (CSS injected)

Run:
pip install streamlit sqlalchemy pandas plotly python-dotenv
streamlit run project_x_expanded.py
"""

import streamlit as st
from sqlalchemy import (create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import io
import os

# ---------------------------
# CONFIG & DB SETUP (SQLAlchemy)
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_expanded.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------------------------
# Grok-style CSS (light version)
# ---------------------------
GROK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
:root{ --bg:#0b0f13; --card:#0f1720; --muted:#93a0ad; --accent:#7c5cff; --accent-2:#00e5a8; --radius:12px; }
body, .stApp { background: linear-gradient(180deg, #06070a 0%, #0b0f13 100%); color: #e6eef6; font-family: 'Inter', sans-serif;}
section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding: 18px; }
.grok-header{ background: linear-gradient(180deg, rgba(124,92,255,0.12), rgba(0,0,0,0.0)); padding: 14px; border-radius:10px; border:1px solid rgba(124,92,255,0.08);}
.grok-panel{ background: var(--card); border-radius: var(--radius); padding:12px; border:1px solid rgba(255,255,255,0.03);}
.metric-card{ background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:10px; border-radius:10px; }
button.stButton>button{ background: linear-gradient(90deg, var(--accent), var(--accent-2)); color:white; border:none; padding:8px 12px; border-radius:8px; }
.input, input, textarea { background: rgba(255,255,255,0.02) !important; color: #e6eef6 !important; border-radius:8px !important; }
"""

# ---------------------------
# MODELS (ORM)
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
# INIT DB
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)

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
        sla_entered_at=datetime.utcnow(),
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
    # update lead status
    lead = est.lead
    lead.status = LeadStatus.AWARDED
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
    session.add(lead)
    session.commit()
    return est

# ---------------------------
# UTIL: convert query results to dataframe
# ---------------------------
def leads_df(session):
    df = pd.read_sql(session.query(Lead).statement, session.bind)
    return df

def estimates_df(session):
    return pd.read_sql(session.query(Estimate).statement, session.bind)

# ---------------------------
# APP UI
# ---------------------------
st.set_page_config(page_title="Project X (ORM)", layout="wide", initial_sidebar_state="expanded")
init_db()
st.markdown(f"<style>{GROK_CSS}</style>", unsafe_allow_html=True)

# Header
st.markdown("<div class='grok-header'><div style=\"font-weight:700; font-size:18px\">Project X — Sales & Conversion (ORM Demo)</div></div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(
            s, source="Google Ads", source_details="gclid=demo", contact_name="Demo", contact_phone="+1-555-0000", contact_email="demo@example.com", property_address="100 Demo Ave", damage_type="water", assigned_to="Alex", estimated_value=1200.0, notes="Demo lead", sla_hours=12, created_by="demo"
        )
        st.success("Demo lead added")
    st.markdown("---")
    st.markdown("App DB: <small>" + DB_FILE + "</small>", unsafe_allow_html=True)

# Page: Leads / Capture
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)")
            contact_name = st.text_input("Contact name")
            contact_phone = st.text_input("Contact phone")
            contact_email = st.text_input("Contact email")
        with col2:
            property_address = st.text_input("Property address")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to")
            estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=50.0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            s = get_session()
            lead = add_lead(
                s, source=source, source_details=source_details, contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email, property_address=property_address, damage_type=damage_type, assigned_to=assigned_to, estimated_value=float(estimated_value) if estimated_value else None, notes=notes, sla_hours=int(sla_hours), created_by="web_user"
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

# Page: Pipeline Board
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
                        entered = lead.sla_entered_at or lead.created_at
                        deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                        remaining = deadline - datetime.utcnow()
                        if remaining.total_seconds() <= 0:
                            st.markdown(f"❗ **SLA OVERDUE** (was due {deadline.strftime('%Y-%m-%d %H:%M')})")
                        else:
                            st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")
                        # stage change
                        new_stage = st.selectbox(f"Move lead #{lead.id} to...", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"stage_{lead.id}")
                        if st.button("Change Stage", key=f"btn_change_{lead.id}"):
                            change_stage(s, lead.id, new_stage, changed_by="ui_user")
                            st.success(f"Lead #{lead.id} moved to {new_stage}")
                            st.rerun()
                        # estimates
                        st.markdown("---")
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

# Page: Analytics & SLA
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

        st.subheader("Conversion by Source")
        conv = df.copy()
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(leads=("id", "count"), awarded=("awarded_flag", "sum")).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        total_leads = len(df)
        awarded = len(df[df.status == LeadStatus.AWARDED])
        st.metric("Total Leads", total_leads)
        st.metric("Awarded (won)", awarded)
        if total_leads > 0:
            st.metric("Overall conversion %", f"{awarded / total_leads * 100:.1f}%")

        st.subheader("SLA / Overdue Leads")
        overdue = []
        for _, row in df.iterrows():
            sla_entered_at = pd.to_datetime(row["sla_entered_at"])
            sla_hours = int(row["sla_hours"]) if pd.notna(row["sla_hours"]) else 24
            deadline = sla_entered_at + pd.to_timedelta(sla_hours, unit="h")
            remaining = deadline - pd.Timestamp.utcnow()
            overdue.append({
                "id": row["id"], "contact": row["contact_name"], "status": row["status"], "sla_stage": row["sla_stage"], "deadline": deadline, "overdue": remaining.total_seconds() <= 0
            })
        df_overdue = pd.DataFrame(overdue)
        if not df_overdue.empty:
            st.dataframe(df_overdue.sort_values("deadline"))
        else:
            st.info("No SLA overdue leads.")

# Page: Exports
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

