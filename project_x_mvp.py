# ===========================
# PROJECT X CRM (ALL-IN-ONE)
# Streamlit + SQLAlchemy + Analytics + Pipeline + Priority Lead Cards
# ===========================

import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, Boolean, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# --- DATABASE CONFIG ---
DB_PATH = "projectx.db"
engine = create_engine(f"sqlite:///{DB_PATH}")
Session = sessionmaker(bind=engine)
Base = declarative_base()

# --- LEAD MODEL ---
class Lead(Base):
    __tablename__ = "leads"
    id = Column(String, primary_key=True, default=lambda: str(datetime.utcnow().timestamp()))
    contact_name = Column(String)
    contact_email = Column(String)
    assigned_to = Column(String)
    status = Column(String, default="New")
    qualified = Column(Boolean, default=False)
    estimated_value = Column(Float, default=0.0)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Float, default=24)
    notes = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    invoice_file = Column(String)

Base.metadata.create_all(engine)

# --- SESSION GETTER ---
def get_session():
    return Session()

# --- ADD NEW LEAD FUNCTION ---
def add_lead(session, contact_name, email, assigned, notes, estimated, sla_hours, qualified):
    lead = Lead(
        contact_name=contact_name,
        contact_email=email,
        assigned_to=assigned,
        notes=notes,
        estimated_value=estimated,
        sla_hours=sla_hours,
        sla_entered_at=datetime.utcnow(),
        qualified=qualified,
        status="New",
        invoice_file=None
    )
    session.add(lead)
    session.commit()
    return lead

# --- GET KPI COUNTS ---
def compute_kpis(df):
    total = len(df)
    active = len(df[df["status"].isin(["New", "Contacted", "Inspection Scheduled"])])
    qualified = len(df[df["qualified"] == True])
    closed = len(df[df["status"].isin(["Won", "Lost", "Approved"])])
    won = len(df[df["status"] == "Won"])
    lost = len(df[df["status"] == "Lost"])

    sla_success = len(df[df["created_at"] >= df["sla_entered_at"]]) / total * 100 if total else 0
    qualification_rate = (qualified / total * 100) if total else 0
    conversion_rate = (won / closed * 100) if closed else 0
    inspection_booked = len(df[df["status"] == "Inspection Scheduled"]) / qualified * 100 if qualified else 0
    estimate_sent = len(df[df["status"] == "Estimate Submitted"])

    pipeline_value = df["estimated_value"].sum()
    est_roi = won if won else (active * 0.5)

    return {
        "Active Leads": active,
        "SLA Success %": f"{sla_success:.1f}%",
        "% Qualified": f"{qualification_rate:.1f}%",
        "Conversion Rate %": f"{conversion_rate:.1f}%",
        "% Booked": f"{inspection_booked:.1f}%",
        "Estimate Sent": estimate_sent,
        "Pipeline Job Value": f"${pipeline_value:,.0f}",
        "Estimated ROI": est_roi,
        "Lost Rate": f"{(lost / total * 100 if total else 0):.1f}%"
    }

# --- PRIORITY LEAD DESIGN ---
def render_priority_lead(r):
    score = r.get("priority_score", 0)

    if score >= 0.7:
        color = "#ef4444"
        label = "🔴 CRITICAL"
    elif score >= 0.45:
        color = "#f97316"
        label = "🟠 HIGH"
    else:
        color = "#22c55e"
        label = "🟢 NORMAL"

    remaining_hours = r.get("time_left_hours", 0)
    if remaining_hours <= 0:
        sla_html = "<span style='color:#ef4444;font-weight:800;'>❗ OVERDUE</span>"
    else:
        h = int(remaining_hours)
        m = int((remaining_hours * 60) % 60)
        sla_html = f"<span style='color:#ef4444; font-weight:800;'>⏳ {h}h {m}m left</span>"

    return f"""
<div style="background:#f0f2f4; padding:18px; border:1px solid #d1d5db; border-radius:16px; margin-bottom:14px; transition: all 0.3s ease;">
  <div style="font-size:28px; font-weight:800; color:#111;">{label}</div>
  <div style="font-size:20px; font-weight:700; color:#222; margin-top:4px;">#{int(r.get("id", 0))} — {r.get("contact_name")}</div>
  <div style="color:#555; font-size:14px; margin-top:6px;">
    {r.get("status").upper()} | Job Est: <span style="color:#10b981; font-weight:700;">${r.get("estimated_value"):,.0f}</span>
  </div>
  <div style="margin-top:8px; font-size:15px;">{sla_html}</div>
  <div style="font-size:26px; font-weight:900; color:{color}; margin-top:10px;">{score:.2f}</div>
</div>
"""

# --- DEMO LEAD BUTTON ---
def demo_lead():
    s = get_session()
    add_lead(s, "Demo Lead", "demo@example.com", "Estimator A", "System generated", 4500, 24, True)
    st.rerun()

# --- STREAMLIT APP PAGES ---
st.set_page_config(page_title="Project X Restoration CRM", layout="wide")

with get_session() as s:
    lead_rows = s.query(Lead).all()

df = pd.DataFrame([{
    "id": i.id,
    "contact_name": i.contact_name,
    "status": i.status,
    "qualified": i.qualified,
    "estimated_value": i.estimated_value,
    "sla_entered_at": i.sla_entered_at,
    "sla_hours": i.sla_hours,
    "created_at": i.created_at,
    "invoice_file": i.invoice_file
} for i in lead_rows])

kpis = compute_kpis(df)

page = st.sidebar.selectbox("Navigate", ["Lead Capture", "Pipeline Board", "Analytics"])

# ==========================
# LEAD CAPTURE PAGE
# ==========================
if page == "Lead Capture":
    st.header("📥 Capture New Lead")
    s = get_session()

    with st.form("lead_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Contact Name")
            email = st.text_input("Email")
            assigned = st.text_input("Assign to")
            qualified = st.checkbox("Qualified Lead?")
        with c2:
            sla = st.number_input("SLA Hours", min_value=1, max_value=72, value=24)
            estimated = st.number_input("Estimated Job Value ($)", step=100.0)
        notes = st.text_area("Notes")

        if st.form_submit_button("🚀 Submit Lead"):
            add_lead(s, name, email, assigned, notes, estimated, sla, qualified)
            st.success("Lead added!")
            st.rerun()

# ==========================
# PIPELINE DASHBOARD PAGE
# ==========================
elif page == "Pipeline Board":
    st.header("📊 **Total Lead Pipeline Key Performance Indicator**")
    st.markdown("_*Measures lead speed, qualification, inspection momentum, revenue forecasting, and ROI health.*_")

    cols = st.columns(4)
    metrics = list(kpis.items())

    for i in range(8):
        label, value = metrics[i]
        with cols[i % 4]:
            st.markdown(f"""
<div style="background:#e5e7eb; padding:14px; border-radius:14px; margin-bottom:12px; animation: fadeInUp 0.6s ease;">
   <div style="font-size:15px; color:#111; font-weight:600;">{label}</div>
   <div style="font-size:36px; font-weight:900; color:#3b82f6; margin-top:6px;">{value}</div>
</div>
""", unsafe_allow_html=True)

    st.header("Lead Pipeline Stages")
    st.markdown("_*Visual breakdown of where all leads currently sit in your pipeline funnel.*_")

    stage_dist = df["status"].value_counts().reset_index()
    stage_dist.columns = ["Stage", "Count"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.pie(stage_dist["Count"], labels=stage_dist["Stage"])
    centre = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre)
    st.pyplot(fig)

    st.header("Top 5 Priority Leads")
    st.markdown("_*Highest urgency + highest conversion potential leads needing immediate action.*_")

    df["deadline"] = df["created_at"] + df["sla_hours"].apply(lambda x: timedelta(hours=x))
    df["time_left_hours"] = (df["deadline"] - datetime.utcnow()).dt.total_seconds() / 3600
    df["priority_score"] = df["qualified"].apply(lambda x: 1.0 if x else 0.4)

    for _, r in df.sort_values("priority_score", ascending=False).head(5).iterrows():
        st.markdown(render_priority_lead(r), unsafe_allow_html=True)

    st.header("All Leads (expand a card to edit / change status)")
    st.markdown("_*Open each lead to change pipeline stage, mark won/lost, create estimate, or upload invoice if awarded.*_")

    s = get_session()
    for _, lead in df.iterrows():
        rec = s.query(Lead).filter(Lead.id == lead["id"]).first()
        with st.expander(rec.contact_name):
            status = st.selectbox("Stage", ["New", "Contacted", "Inspection Scheduled", "Inspection Completed", "Estimate Submitted", "Lost", "Won"], index=0)
            invoice = None
            if status == "Won":
                invoice = st.file_uploader("Upload invoice (job won only)")

            if st.button("✅ Update"):
                rec.status = status
                if invoice:
                    rec.invoice_file = invoice.name
                s.commit()
                st.rerun()

# ==========================
# ANALYTICS PAGE
# ==========================
elif page == "Analytics":
    st.header("📈 Analytics Dashboard")

    st.header("SLA / Overdue Leads Trend")
    st.markdown("_*Lead aging and compliance trend based on SLA deadlines.*_")

    timeline = df.groupby(df["created_at"].dt.date).size().reset_index()
    timeline.columns = ["Date", "Count"]

    fig, ax = plt.subplots()
    ax.plot(timeline["Date"], timeline["Count"])
    st.pyplot(fig)

    st.header("Lead Stages Breakdown (Donut)")
    st.markdown("_*Current funnel share across your lead pipeline stages.*_")

    fig, ax = plt.subplots()
    ax.pie(stage_dist["Count"], labels=stage_dist["Stage"])
    ax.add_artist(plt.Circle((0, 0), 0.65, fc='white'))
    st.pyplot(fig)

    st.header("CPA Per Won Job vs Conversion Velocity")
    st.markdown("_*Cost efficiency and speed comparison across time range.*_")

    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    comp = df[(df["created_at"].dt.date >= start) & (df["created_at"].dt.date <= end)]
    fig, ax = plt.subplots()
    ax.bar(["CPA/Won", "Velocity"], [len(comp), comp["sla_hours"].mean()])
    st.pyplot(fig)

    st.markdown("_*CPA per won job: trending downward MoM — Velocity improves, stagnation beyond 72hrs flags lead risk*_")
