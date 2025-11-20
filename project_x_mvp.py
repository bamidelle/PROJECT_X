# project_x_mvp.py
"""
Project X - MVP (Step 1)
Lead Capture + Pipeline + Analytics + SLA countdown + CSV export
Single-file Streamlit app using SQLite (SQLAlchemy) for beginners.

Run:
pip install streamlit pandas plotly sqlalchemy
streamlit run project_x_mvp.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import enum
import io
import sqlite3
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# Config
# ---------------------------
DB_FILE = "project_x_mvp.db"
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


# ---------------------------
# Models (SQLAlchemy) - simple and readable for a beginner
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
    source = Column(String, default="Unknown")       # e.g., Google, Referral, Phone
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)      # water, fire, mold...
    status = Column(String, default=LeadStatus.NEW)
    assigned_to = Column(String, nullable=True)
    estimated_value = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    sla_hours = Column(Integer, default=24)          # default SLA for first response
    sla_stage = Column(String, default=LeadStatus.NEW)  # which stage the SLA applies to
    sla_entered_at = Column(DateTime, default=datetime.utcnow)  # timestamp when current SLA started

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
# DB Init
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)


# ---------------------------
# Utility functions
# ---------------------------
def get_session():
    return SessionLocal()


def add_lead(session, **kwargs):
    # create lead and initial stage history
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
    # stage history
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
    # when stage changes, reset SLA timers to the new stage if desired
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
    # also update lead status to Awarded
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
    # update lead to Lost
    lead = est.lead
    lead.status = LeadStatus.LOST
    session.add(lead)
    session.commit()
    return est


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Project X — Sales & Conversion MVP", layout="wide")
init_db()

st.title("🚀 Project X — Sales & Conversion Tracker (MVP)")
st.markdown("**Step 1:** Lead capture → pipeline → funnel analytics. Beginner friendly. Save data locally in SQLite.")

# Layout: sidebar for quick actions, main area for pages
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")
    st.markdown("Quick add demo lead")
    if st.button("Add Demo Lead"):
        session = get_session()
        add_lead(
            session,
            source="Google Ads",
            source_details="gclid=demo",
            contact_name="Demo Customer",
            contact_phone="+1-555-0100",
            contact_email="demo@example.com",
            property_address="123 Demo St",
            damage_type="water",
            assigned_to="Alex",
            estimated_value=1250.0,
            notes="Demo lead created from sidebar quick action.",
            sla_hours=12,
            created_by="demo"
        )
        st.success("Demo lead added")
    st.markdown("---")
    st.markdown("Tips:")
    st.markdown("- Start by adding leads in **Leads / Capture**")
    st.markdown("- Move leads through the pipeline in **Pipeline Board**")
    st.markdown("- View SLA overdue in **Analytics & SLA**")
    st.markdown("---")
    st.markdown("App stores data in local file: `project_x_mvp.db`")


# ---------------------------
# Page: Leads / Capture
# ---------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    st.markdown("Fill the form to create a new lead (this writes to local SQLite).")

    with st.form("lead_form"):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source Details (UTM/notes)")
            contact_name = st.text_input("Contact Name")
            contact_phone = st.text_input("Contact Phone")
            contact_email = st.text_input("Contact Email")
        with col2:
            property_address = st.text_input("Property Address")
            damage_type = st.selectbox("Damage Type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned To (salesperson)")
            estimated_value = st.number_input("Estimated Job Value (USD)", min_value=0.0, value=0.0, step=50.0)
            sla_hours = st.number_input("SLA (hours for first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes / Additional Info")
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            session = get_session()
            lead = add_lead(
                session,
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
    st.subheader("Recent Leads")
    session = get_session()
    leads_q = session.query(Lead).order_by(Lead.created_at.desc()).limit(25).all()
    if leads_q:
        df = pd.DataFrame([{
            "id": l.id,
            "created_at": l.created_at,
            "source": l.source,
            "contact": l.contact_name,
            "phone": l.contact_phone,
            "damage_type": l.damage_type,
            "status": l.status,
            "assigned_to": l.assigned_to,
            "est_value": l.estimated_value
        } for l in leads_q])
        st.dataframe(df)
    else:
        st.info("No leads yet. Create one with the form above.")


# ---------------------------
# Page: Pipeline Board
# ---------------------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board")
    st.markdown("View leads grouped by stage and change stage to progress the lead.")

    session = get_session()
    all_leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    if not all_leads:
        st.info("No leads yet. Add leads on the Lead Capture page.")
    else:
        # Group leads by status
        cols = st.columns(len(LeadStatus.ALL))
        status_columns = LeadStatus.ALL
        # Prepare simple dict of lists
        buckets = {s: [] for s in status_columns}
        for l in all_leads:
            buckets[l.status].append(l)

        # Display each column
        for i, s in enumerate(status_columns):
            with cols[i]:
                st.markdown(f"### {s} ({len(buckets[s])})")
                for lead in buckets[s]:
                    with st.expander(f"#{lead.id} · {lead.contact_name or 'No name'} · ${lead.estimated_value or 0:,.0f}"):
                        st.write(f"**Source:** {lead.source} · **Assigned:** {lead.assigned_to}")
                        st.write(f"**Address:** {lead.property_address}")
                        st.write(f"**Damage:** {lead.damage_type}")
                        st.write(f"**Notes:** {lead.notes}")
                        st.write(f"**Created:** {lead.created_at}")
                        # Show SLA remaining
                        entered = lead.sla_entered_at or lead.created_at
                        deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                        remaining = deadline - datetime.utcnow()
                        if remaining.total_seconds() <= 0:
                            st.markdown(f"❗ **SLA OVERDUE** (was due {deadline.strftime('%Y-%m-%d %H:%M')})")
                        else:
                            st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                        # Stage change UI
                        new_stage = st.selectbox(f"Move lead #{lead.id} to...", options=LeadStatus.ALL, key=f"stage_{lead.id}")
                        if st.button("Change Stage", key=f"btn_change_{lead.id}"):
                            change_stage(session, lead.id, new_stage, changed_by="ui_user")
                            st.success(f"Lead #{lead.id} moved to {new_stage}")
                            st.rerun()

                        # Create Estimate
                        st.markdown("---")
                        st.write("Estimates")
                        ests = session.query(Estimate).filter(Estimate.lead_id == lead.id).all()
                        if ests:
                            est_df = pd.DataFrame([{
                                "id": e.id,
                                "amount": e.amount,
                                "sent_at": e.sent_at,
                                "approved": e.approved,
                                "lost": e.lost,
                                "lost_reason": e.lost_reason
                            } for e in ests])
                            st.dataframe(est_df)
                        else:
                            st.write("No estimates yet.")

                        with st.form(f"est_form_{lead.id}", clear_on_submit=True):
                            amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"est_amt_{lead.id}")
                            details = st.text_area("Estimate details", key=f"est_det_{lead.id}")
                            if st.form_submit_button("Create Estimate", key=f"est_submit_{lead.id}"):
                                create_estimate(session, lead.id, float(amt), details=details)
                                st.success("Estimate created")
                                st.rerun()

                        # Buttons to mark estimate actions (choose first estimate for demo)
                        if ests:
                            first_est = ests[0]
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                if st.button("Mark Sent", key=f"send_{first_est.id}"):
                                    mark_estimate_sent(session, first_est.id)
                                    st.success("Marked as sent")
                                    st.rerun()
                            with col_b:
                                if st.button("Mark Approved", key=f"app_{first_est.id}"):
                                    mark_estimate_approved(session, first_est.id)
                                    st.success("Estimate approved and lead marked Awarded")
                                    st.rerun()
                            with col_c:
                                if st.button("Mark Lost", key=f"lost_{first_est.id}"):
                                    mark_estimate_lost(session, first_est.id, reason="Lost to competitor")
                                    st.success("Estimate marked lost and lead moved to Lost")
                                    st.rerun()


# ---------------------------
# Page: Analytics & SLA
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    session = get_session()
    df_leads = pd.read_sql(session.query(Lead).statement, session.bind)
    if df_leads.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Funnel counts
        funnel = df_leads.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
        funnel.columns = ["stage", "count"]

        st.subheader("Funnel Overview")
        fig = px.bar(funnel, x="stage", y="count", title="Leads by Stage", text="count")
        fig.update_layout(xaxis_title=None, yaxis_title="Number of Leads", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        # Conversion by source
        st.subheader("Conversion by Source")
        conv = df_leads.copy()
        # mark awarded as 1 else 0
        conv["awarded_flag"] = conv["status"].apply(lambda x: 1 if x == LeadStatus.AWARDED else 0)
        conv_summary = conv.groupby("source").agg(
            leads=("id", "count"),
            awarded=("awarded_flag", "sum")
        ).reset_index()
        conv_summary["conversion_rate"] = (conv_summary["awarded"] / conv_summary["leads"] * 100).round(1)
        st.dataframe(conv_summary.sort_values("leads", ascending=False))

        # Simple conversion funnel percentages
        total_leads = len(df_leads)
        awarded = len(df_leads[df_leads.status == LeadStatus.AWARDED])
        st.metric("Total Leads", total_leads)
        st.metric("Awarded (won)", awarded)
        if total_leads > 0:
            st.metric("Overall conversion %", f"{awarded / total_leads * 100:.1f}%")

        # SLA overview: list overdue leads
        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df_leads.iterrows():
            sla_entered_at = pd.to_datetime(row["sla_entered_at"])
            sla_hours = int(row["sla_hours"]) if pd.notna(row["sla_hours"]) else 24
            deadline = sla_entered_at + pd.to_timedelta(sla_hours, unit="h")
            remaining = deadline - pd.Timestamp.utcnow()
            overdue = remaining.total_seconds() <= 0
            overdue_rows.append({
                "id": row["id"],
                "contact": row["contact_name"],
                "status": row["status"],
                "sla_stage": row["sla_stage"],
                "deadline": deadline,
                "overdue": overdue
            })
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.dataframe(df_overdue.sort_values("deadline"))
            st.markdown("**Overdue leads are flagged.** Contact them immediately to improve conversion.")
        else:
            st.info("No SLA overdue leads.")

# ---------------------------
# Page: Exports
# ---------------------------
elif page == "Exports":
    st.header("📤 Export data")
    session = get_session()
    df_leads = pd.read_sql(session.query(Lead).statement, session.bind)
    df_hist = pd.read_sql(session.query(StageHistory).statement, session.bind)
    df_est = pd.read_sql(session.query(Estimate).statement, session.bind)

    st.subheader("Download CSVs")
    if not df_leads.empty:
        csv_leads = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv_leads, file_name="leads.csv", mime="text/csv")
    else:
        st.info("No leads available for export.")

    if not df_est.empty:
        csv_est = df_est.to_csv(index=False).encode("utf-8")
        st.download_button("Download estimates.csv", csv_est, file_name="estimates.csv", mime="text/csv")
    if not df_hist.empty:
        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button("Download stage_history.csv", csv_hist, file_name="stage_history.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("Tip: Use CSV exports to import data into your accounting system or to train ML models later.")

# End of app
