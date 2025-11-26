# =======================
#  PROJECT X — MERGED SINGLE FILE APP
# =======================

import streamlit as st
import os, joblib
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import matplotlib.pyplot as plt

# --- App Page Config ---
st.set_page_config(page_title="Pipeline Board", layout="wide")

# --- UI Styling ---
st.markdown("""
<style>
body, .stApp { background:#fff; font-family: Comfortaa; }
.metric-card {
    background:#111;
    border-radius:14px;
    padding:18px;
    margin:10px;
    color:white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn {
    0%{opacity:0; transform:translateY(6px)}
    100%{opacity:1; transform:translateY(0)}
}
.stage-badge {
    padding:4px 12px;
    border-radius:12px;
    font-size:11px;
    font-weight:700;
    margin-left:8px;
}
.glassy-bg {
    background:rgba(0,0,0,0.85);
    backdrop-filter: blur(10px);
    border-radius:16px;
    padding:22px;
    color:white;
}
.main-btn button {
    width:100%;
    padding:10px 24px;
    border-radius:10px;
    font-weight:700;
    font-size:15px;
    animation: pulse 1.5s infinite alternate;
}
@keyframes pulse {
    0%{transform:scale(1)}
    100%{transform:scale(1.03)}
}
</style>
""", unsafe_allow_html=True)

# --- Database Setup ---
Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}")
SessionLocal = sessionmaker(bind=engine)

class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]

STATUS_COLORS = {
    "New": "#2563eb",
    "Contacted": "#eab308",
    "Inspection Scheduled": "#f97316",
    "Inspection Completed": "#14b8a6",
    "Estimate Submitted": "#a855f7",
    "Awarded": "#22c55e",
    "Lost": "#ef4444"
}

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    assigned_to = Column(String)
    estimated_value = Column(Float, default=0)
    notes = Column(String)
    status = Column(String, default="New")
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    invoice_file = Column(String)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float)
    sent_at = Column(DateTime)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def add_lead(session, **kwargs):
    lead = Lead(**kwargs)
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead

def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount)
    session.add(est)
    session.commit()
    return est

def mark_estimate_sent(session, est_id):
    est = session.get(Estimate, est_id)
    if est:
        est.sent_at = datetime.utcnow()
        session.commit()

def compute_priority_for_lead_row(row):
    score = 1.0
    val = float(row.get("estimated_value") or 0)
    if val < 500: score = 0.4
    if val > 5000: score = 0.8
    if row.get("status") == "Awarded": score = 1.0
    if row.get("status") == "Lost": score = 0.1
    return score

# --- Ensure DB is initialized ---
init_db()

# --- Page Routing ---
page = st.sidebar.selectbox("Navigation", ["Lead Capture", "Pipeline Board", "Analytics"])

# --- Pipeline Board ---
if page == "Pipeline Board":
    st.header("📊 Pipeline Dashboard (Google Ads style)")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if leads:
        df = leads_df = pd.DataFrame([{
            "id": l.id,
            "contact_name": l.contact_name,
            "contact_phone": l.contact_phone,
            "contact_email": l.contact_email,
            "property_address": l.property_address,
            "damage_type": l.damage_type,
            "assigned_to": l.assigned_to,
            "estimated_value": l.estimated_value,
            "notes": l.notes,
            "status": l.status,
            "sla_hours": l.sla_hours,
            "sla_entered_at": l.sla_entered_at,
            "created_at": l.created_at,
            "qualified": l.qualified
        } for l in leads])

        # KPI summary 2x4 grid
        funnel_counts = df['status'].value_counts().to_dict()
        kpi_items = []
        for stage in LeadStatus.ALL:
            count = funnel_counts.get(stage, 0)
            clr = STATUS_COLORS.get(stage, "#000")
            kpi_items.append((stage, count, clr))
        
        st.markdown("<div class='glassy-bg'>", unsafe_allow_html=True)
        rows = [st.columns(4), st.columns(4)]
        for i, (stage, cnt, clr) in enumerate(kpi_items):
            r = 0 if i < 4 else 1
            with rows[r][i % 4]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color:white;font-size:15px;font-weight:700;">{stage}</div>
                    <div style="color:{clr};font-size:36px;font-weight:800;margin-top:6px;">{cnt}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="background:{clr};width:{(cnt/max(1,len(df))*100)}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Priority leads
        st.markdown("### 🎯 Priority Leads (Top 8)")
        if "pipeline_stage_filter" not in st.session_state:
            st.session_state.pipeline_stage_filter = None

        # SLA formatting helper
        def sla_hms(dl):
            rem = dl - datetime.utcnow()
            if rem.total_seconds() <= 0:
                return "<span style='color:#ef4444;font-weight:800;'>⚠ OVERDUE</span>"
            h = int(rem.total_seconds()//3600)
            m = int((rem.total_seconds()%3600)//60)
            return f"<span style='color:#ef4444;font-weight:700;'>{h}h {m}m left</span>"

        pr_list = []
        for _, row in leads_df.iterrows():
            score = compute_priority_for_lead_row(row)
            l = s.get(Lead, int(row["id"]))
            entered = l.sla_entered_at or l.created_at
            dl = entered + timedelta(hours=l.sla_hours)
            pr_list.append({
                "id": l.id,
                "contact_name": l.contact_name,
                "estimated_value": l.estimated_value or 0,
                "priority_score": score,
                "damage_type": l.damage_type,
                "status": l.status,
                "deadline": dl,
                "sla_html": sla_hms(dl),
                "conversion_prob": (joblib.load("lead_conversion_model.pkl").predict_proba([[l.estimated_value,1,1,1]])[0][1]
                                   if os.path.exists("lead_conversion_model.pkl") else None),
                "invoice_file": l.invoice_file
            })

        pr_df = pd.DataFrame(pr_list).sort_values("priority_score", ascending=False)
        for _, r in pr_df.head(8).iterrows():
            st.markdown(render_priority_lead_card(r, STATUS_COLORS), unsafe_allow_html=True)

        # All leads editable
        st.markdown("### 📋 All Leads (Expand to Edit)")
        for lead in leads:
            with st.expander(f"#{lead.id} — {lead.contact_name or 'No name'}"):
                st.write(f"Address: {lead.property_address}")
                st.write(f"Job Value Estimate: {lead.estimated_value}")

                # Editable status
                new_status = st.selectbox("Change Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")
                if new_status == "Awarded":
                    inv = st.file_uploader("📤 Upload Invoice (optional)", key=f"inv_{lead.id}")
                    if inv:
                        path = save_uploaded = f"uploaded_invoices/lead_{lead.id}_{int(datetime.utcnow().timestamp())}.pdf"
                        os.makedirs("uploaded_invoices", exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(inv.getbuffer())
                        lead.invoice_file = path

                if st.button("💾 Save Progress", key=f"save_{lead.id}"):
                    lead.status = new_status
                    lead.sla_entered_at = datetime.utcnow()
                    s.add(lead)
                    s.commit()
                    st.success("Updated!")
                    st.rerun()

    else:
        st.info("No leads available")

# --- Lead Capture ---
elif page == "Lead Capture":
    st.header("➕ Create a Lead")

    with st.form("lead_form"):
        name = st.text_input("Contact Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        addr = st.text_input("Property Address")
        dmg = st.selectbox("Damage Type", ["Water", "Fire", "Mold", "Reconstruction"])
        val = st.number_input("Job Value Estimate (USD)", min_value=0.0, step=100.0)
        notes = st.text_area("Notes")

        if st.form_submit_button("🚀 Capture Lead"):
            s = get_session()
            add_lead(s,
                contact_name=name, contact_phone=phone, contact_email=email,
                property_address=addr, damage_type=dmg,
                assigned_to="", estimated_value=val,
                notes=notes, status="New", qualified=True,
                sla_entered_at=datetime.utcnow()
            )
            st.success("Lead Captured!")
            st.rerun()

# --- Analytics ---
elif page == "Analytics":
    st.header("📈 Analytics (Auto updating Pie Chart)")

    s = get_session()
    leads = s.query(Lead).all()
    if leads:
        df = leads_df = pd.DataFrame([{"status":l.status} for l in leads])
        fig, ax = plt.subplots()
        ax.pie(df['status'].value_counts(), labels=df['status'].value_counts().index, autopct='%1.1f%%')
        st.pyplot(fig)
    else:
        st.info("No analytics yet.")
