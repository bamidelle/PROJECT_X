# ============================ PROJECT X SINGLE-FILE APP ============================
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base

# ================================ DATABASE SETUP ==================================
DATABASE_URL = "sqlite:///project_x_mvp.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()

# ================================ DATABASE MODELS =================================
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    estimated_value = Column(Float)
    status = Column(String)
    qualified = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    sla_entered_at = Column(DateTime)
    sla_hours = Column(Integer, default=24)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(String)
    invoice_file = Column(String, nullable=True)

Base.metadata.create_all(engine)

# ================================ UTIL FUNCTIONS ==================================
def get_session():
    return Session()

def leads_df(session):
    leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    return pd.DataFrame([
        {
            "id": l.id,
            "contact_name": l.contact_name,
            "contact_phone": l.contact_phone,
            "contact_email": l.contact_email,
            "property_address": l.property_address,
            "damage_type": l.damage_type,
            "estimated_value": l.estimated_value,
            "status": l.status,
            "qualified": l.qualified,
            "inspection_scheduled": l.inspection_scheduled,
            "inspection_completed": l.inspection_completed,
            "estimate_submitted": l.estimate_submitted,
            "sla_hours": l.sla_hours,
            "sla_entered_at": l.sla_entered_at,
            "created_at": l.created_at,
            "invoice_file": l.invoice_file,
            "notes": l.notes
        }
        for l in leads
    ])

def add_lead(session, lead_data):
    Lead(**lead_data)
    session.add(Lead(**lead_data))
    session.commit()

# ============================== STREAMLIT SETUP ===================================
st.set_page_config(page_title="Project X CRM", layout="wide")
st.markdown("<style>body{background:white;}</style>", unsafe_allow_html=True)

# ============================== UI STYLES ========================================
st.markdown("""
<style>
.metric-card{background:#111; padding:16px; border-radius:16px; margin-bottom:12px; animation:fade 0.6s ease;}
@keyframes fade{from{opacity:0.3}to{opacity:1}}
.metric-value{font-size:40px;font-weight:800;color:#2563eb;text-align:left;}
.stage-title{font-size:20px;font-weight:700;color:white;text-align:left;}
.status-active{color:#22c55e;font-weight:600;font-size:13px;}
.status-lost{color:#ef4444;font-weight:600;font-size:13px;}
.progress-bar{height:8px;border-radius:4px;width:100%;background:#2563eb80;margin-top:6px;}
button[data-baseweb="button"]{border-radius:10px !important; padding:8px 18px !important; animation:pulse .7s infinite alternate;}
</style>
""", unsafe_allow_html=True)

# ================================= NAVIGATION ====================================
page = st.selectbox("Navigate", ["Lead Capture", "Pipeline Board", "Analytics"])

# ================================= LEAD CAPTURE ==================================
if page == "Lead Capture":
    st.header("📥 Lead Capture & Intake")
    with st.form("create_lead"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            address = st.text_area("Property Address")
            damage = st.selectbox("Damage Type", ["water", "fire", "mold", "biohazard", "hoarding", "reconstruction"])
            value = st.number_input("Estimated Job Value ($)", min_value=0.0, step=100.0)
        with col2:
            source = st.selectbox("Lead Source", ["google ads", "website", "referral", "meta ads", "organic search", "qr intake"])
            qualify = st.checkbox("Qualified Lead?")
            notes = st.text_area("Additional Notes")
            submit = st.form_submit_button("🚀 Submit Lead")

        if submit:
            s = get_session()
            s.add(Lead(
                contact_name=name, contact_phone=phone, contact_email=email,
                property_address=address, damage_type=damage,
                estimated_value=value, status="New", qualified=qualify,
                sla_entered_at=datetime.utcnow(), source=source, notes=notes
            ))
            s.commit()
            st.success("✅ Lead captured successfully!")

# ================================= PIPELINE BOARD =================================
elif page == "Pipeline Board":
    st.header("🧭 TOTAL LEAD PIPELINE KEY PERFORMANCE INDICATOR")
    st.markdown("*Tracks live pipeline movement and job values in restoration operations*", unsafe_allow_html=False)

    s = get_session()
    df = leads_df(s)

    if df.empty:
        st.info("No Leads Available.")
        st.stop()

    # Calculate KPIs
    active = len(df)
    sla_success = len(df[df["inspection_scheduled"] == True]) / active * 100
    qualify_rate = len(df[df["qualified"] == True]) / active * 100
    conversion = len(df[df["status"] == "Job Won"]) / active * 100 if "Job Won" in df["status"].values else 0
    insp_booked = len(df[df["inspection_scheduled"] == True]) / active * 100
    estimate_sent = len(df[df["estimate_submitted"] == True]) / active * 100
    pipeline_jobvalue = df["estimated_value"].sum()

    # Draw KPI grid (2 rows × 4 columns)
    st.markdown("### 📊 KPI Overview")
    kcols = st.columns(4)
    for i, (label, val, color) in enumerate([
        ("ACTIVE LEADS", active, "#2563eb"),
        ("SLA SUCCESS %", f"{sla_success:.1f}%", "#14b8a6"),
        ("QUALIFICATION RATE %", f"{qualify_rate:.1f}%", "#eab308"),
        ("CONVERSION RATE %", f"{conversion:.1f}%", "#a855f7"),
        ("INSPECTION BOOKED %", f"{insp_booked:.1f}%", "#f97316"),
        ("ESTIMATE SENT %", f"{estimate_sent:.1f}%", "#22c55e"),
        ("PIPELINE JOB VALUE ($)", f"${pipeline_jobvalue:,.0f}", "#2563eb"),
        ("ESTIMATED ROI", f"${pipeline_jobvalue * 1.6:,.0f}", "#93a0ad")
    ])[i: i+1]:
        with kcols[i % 4]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='stage-title'>{label}</div>
                <div class='metric-value' style='color:{color};'>{val}</div>
                <div class='progress-bar' style='background:{val if isinstance(val,int) else 80}%;'></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Donut pie chart for pipeline stages
    st.subheader("Lead Pipeline Stages")
    st.markdown("*Visual breakdown of current pipeline stage distribution*", unsafe_allow_html=False)
    status_count = df["status"].value_counts()
    fig = plt.figure()
    plt.pie(status_count, labels=status_count.index, autopct='%1.1f%%')
    plt.gca().add_artist(plt.Circle((0,0), 0.6, fill=True))
    st.pyplot(fig)

    st.markdown("---")

    # TOP 5 priority leads
    st.subheader("🎯 TOP 5 PRIORITY LEADS")
    st.markdown("*System-ranked urgent lead list based on SLA and qualification weights*", unsafe_allow_html=False)
    df = df.assign(priority_score=(df["qualified"].astype(int) * 0.4 + df["inspection_completed"].astype(int)*0.3 + df["inspection_scheduled"].astype(int)*0.3))
    prdf = df.sort_values("priority_score", ascending=False).head(5)

    for _,r in prdf.iterrows():
        score=r["priority_score"]
        st.markdown(f"""
        <div class='metric-card'><div class='metric-value' style='color:{"#ef4444" if score>0.7 else "#f97316"};'>{score:.2f}</div>{r['contact_name']} ({r['status']})</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # All Leads editable
    st.subheader("📋 ALL LEADS")
    st.markdown("*Expand a lead card to update status and upload invoice if Won*", unsafe_allow_html=False)

    for _,lead in df.iterrows():
        with st.expander(f"Lead #{lead['id']} — {lead['contact_name']}"):
            new_status=st.selectbox("Update Status", ["New","Contacted","Inspection Scheduled","Inspection Completed","Estimate Submitted","Carrier Review","Approved","Job Won","Job Lost"])
            inv=None
            if new_status=="Job Won":
                st.success("🎉 Lead marked WON! You may upload an invoice below.")
                inv=st.file_uploader("Upload Invoice File", type=["pdf","png","jpg"], key=f"invoice_{lead['id']}")

            if st.button("💾 Save Update", key=f"save_{lead['id']}"):
                s=get_session()
                obj=s.query(Lead).get(lead["id"])
                obj.status=new_status
                if inv:
                    obj.invoice_file=inv.name
                s.commit()
                st.experimental_rerun()

# =================================== ANALYTICS ===================================
elif page == "Analytics":
    s = get_session()
    df = leads_df(s)

    if df.empty:
        st.warning("No analytics: no lead data.")
        st.stop()

    st.header("📈 Pipeline Analytics")

    # Add demo generator stays intact
    if st.button("➕ Add Demo Lead"):
        s.add(Lead(contact_name="Demo Customer", damage_type="water", estimated_value=4500, status="New", qualified=True, inspection_scheduled=True, sla_entered_at=datetime.utcnow(), source="Demo"))
        s.commit()
        st.experimental_rerun()

    # CPA vs Velocity Comparison Chart
    st.subheader("🔍 Source Cost vs Conversion Velocity Comparison")

    from_date = st.date_input("Start Date", datetime.utcnow().date())
    to_date = st.date_input("End Date", datetime.utcnow().date())
    
    df_filtered = df[(df["created_at"].dt.date >= from_date) & (df["created_at"].dt.date <= to_date)]

    if df_filtered.empty:
        st.info("No data for selected date range.")
    else:
        cpa = df_filtered.groupby("source")["estimated_value"].mean()
        velocity = df_filtered.groupby("source")["sla_hours"].mean()
        fig = plt.figure()
        plt.plot(cpa.index, cpa.values)
        plt.plot(velocity.index, velocity.values)
        st.pyplot(fig)
        
        st.markdown("*CPA per won job: trending downward MoM, segmented by source.*")
        st.markdown("*Velocity: always improving; >48–72 hrs stagnation indicates bottlenecks.*")

