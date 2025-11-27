import os
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# ================= DATABASE SETUP =================
Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ================= MODELS =================

class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION = "Inspection Scheduled"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION, ESTIMATE_SUBMITTED, AWARDED, LOST]

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    assigned_to = Column(String)
    estimated_value = Column(Float)
    notes = Column(String)
    status = Column(String, default=LeadStatus.NEW)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    invoice_file = Column(String, nullable=True)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float)
    sent_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def leads_df(session):
    try:
        leads = session.query(Lead).all()
        data = [{
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
        } for l in leads]
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Database Query Failed: {e}")
        return pd.DataFrame()

def add_demo_data(session):
    if session.query(Lead).count() == 0:
        demo_leads = [
            Lead(contact_name="Demo Customer", contact_phone="08000000000", contact_email="demo@mail.com",
                 property_address="123 Demo St", damage_type="Water", assigned_to="Estimator A",
                 estimated_value=4500, notes="Demo notes", qualified=True),

            Lead(contact_name="Tee James", contact_phone="09011112222", contact_email="tee@mail.com",
                 property_address="45 Sample Ave", damage_type="Reconstruction", assigned_to="Estimator B",
                 estimated_value=5600, notes="Priority lead test", qualified=True)
        ]
        session.add_all(demo_leads)
        session.commit()

def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount)
    session.add(est)
    session.commit()
    lead = session.query(Lead).get(lead_id)
    if lead:
        lead.estimated_value = amount
        lead.status = LeadStatus.ESTIMATE_SUBMITTED
        lead.sla_entered_at = datetime.utcnow()
        session.commit()

def mark_estimate_sent(session, est_id):
    est = session.query(Estimate).get(est_id)
    if est:
        est.sent_at = datetime.utcnow()
        session.commit()

def mark_estimate_approved(session, est_id):
    est = session.query(Estimate).get(est_id)
    if est:
        est.approved = True
        lead = session.query(Lead).get(est.lead_id)
        if lead:
            lead.status = LeadStatus.AWARDED
            lead.created_at = datetime.utcnow()
        session.commit()

def mark_estimate_lost(session, est_id, reason=None):
    est = session.query(Estimate).get(est_id)
    if est:
        est.lost = True
        lead = session.query(Lead).get(est.lead_id)
        if lead:
            lead.status = LeadStatus.LOST
        session.commit()

# ============== PRIORITY LOGIC ================

def compute_priority_for_lead_row(lead_row, weights):
    val = float(lead_row.get("estimated_value") or 0)
    baseline = weights.get("value_baseline", 5000.0)
    value_score = min(val / baseline, 1.0)

    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at") or datetime.utcnow()
    if isinstance(sla_entered, str):
        try: sla_entered = datetime.fromisoformat(sla_entered)
        except: sla_entered = datetime.utcnow()

    deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
    remaining_h = max((deadline - datetime.utcnow()).total_seconds() / 3600, 0.0)
    sla_score = 1.0 if deadline < datetime.utcnow() else max(0.0, (72 - min(remaining_h, 72)) / 72)

    urgency = weights.get("urgency_weight", 0.15)
    total_weight = weights.get("value_weight", 0.5) + weights.get("sla_weight", 0.35) + urgency
    total_weight = total_weight if total_weight > 0 else 1.0

    score = (value_score * weights.get("value_weight", 0.5) +
             sla_score * weights.get("sla_weight", 0.35)) / total_weight

    return round(max(0.0, min(score, 1.0)), 3), None, None, None, None, None, remaining_h

# =========================== UI ===========================

st.set_page_config(page_title="Project X", layout="wide")

init_db()
s = get_session()
add_demo_data(s)

st.markdown("""
<style>
body, .stApp {background:white;}
button {transition:0.3s ease; border-radius:8px; padding: 10px 18px;}
button:hover {transform:scale(1.03);}
.main-submit-btn {
    background:#3b82f6; color:#fff; font-size:16px; font-weight:600;
    padding:10px 20px; border-radius:10px; border:none;
    animation:pulse 1.8s infinite;
}
@keyframes pulse {
  0%{box-shadow:0 0 0 0 rgba(59,130,246,0.35);}
  70%{box-shadow:0 0 0 8px rgba(59,130,246,0);}
  100%{box-shadow:0 0 0 0 rgba(59,130,246,0);}
}
.metric-card-dark {
    background:#111; color:white; padding:18px; border-radius:12px; margin-bottom:14px;
    border:1px solid #333;
}
.metric-number {font-size:36px; font-weight:800; margin-top:6px;}
.metric-title {font-size:14px; color:#e5e7eb; font-weight:600;}
.progress-grey {width:100%; background:#e5e7eb; height:8px; border-radius:6px; overflow:hidden; margin-top:8px;}
</style>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation",
    ["Lead Capture", "Pipeline Board", "Analytics"])

if menu == "Lead Capture":
    st.header("📥 Lead Capture")
    with st.form("capture"):
        contact_name = st.text_input("Contact Name")
        contact_phone = st.text_input("Phone")
        contact_email = st.text_input("Email")
        property_address = st.text_input("Property Address")
        damage_type = st.selectbox("Damage Type", ["Water","Fire","Reconstruction","Mold"])
        assigned_to = st.text_input("Assigned To")
        estimated_value = st.number_input("Estimated Job Value ($)", min_value=0.0, step=100.0)
        notes = st.text_area("Notes")
        if st.form_submit_button("Submit", type="primary"):
            s = get_session()
            add_lead(s,
                contact_name=contact_name, contact_phone=contact_phone, contact_email=contact_email,
                property_address=property_address, damage_type=damage_type,
                assigned_to=assigned_to, estimated_value=estimated_value, notes=notes,
                sla_entered_at=datetime.utcnow(), qualified=True)
            st.success("Lead saved ✅")

elif menu == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard")

    df = leads_df(s)
    total_leads = len(df)

    kpi_list = []
    for _, row in df.iterrows():
        score, _, _, _, _, _, time_left_h = compute_priority_for_lead_row(row, st.session_state.get("weights", {}))
        sla_s, overdue = compute_priority_for_lead_row(row, st.session_state.get("weights", {}))[0:2]
        kpi_list.append({"id":row["id"], "contact_name":row["contact_name"], "status":row["status"],
                         "estimated_value":row["estimated_value"], "priority_score":score,
                         "time_left_hours":time_left_h, "damage_type":row["damage_type"]})

    pr_df = pd.DataFrame(kpi_list).sort_values("priority_score", ascending=False)

    # ====== KPI 4x4 GRID ======
    kpiA = len(df[df['status']=="New"])
    kpiB = len(df[df['qualified']==True])
    kpiC = len(df[df['status']=="Inspection Scheduled"])
    kpiD = len(df[df['status']=="Estimate Submitted"])
    kpiE = len(df[df['status']=="Awarded"])
    kpiF = len(df[df['status']=="Lost"])
    kpiG = df['estimated_value'].sum()
    kpiH = (kpiE / (kpiE + kpiF) * 100) if (kpiE + kpiF) else 0

    grid = st.columns(4)
    metrics = [
        ("Total Leads", total_leads, "#3b82f6"),
        ("Lead Compliance", kpiA, "#2563eb"),
        ("Qualification %", f"{kpiB} ({kpiB/max(total_leads,1)*100:.0f}%)", "#eab308"),
        ("Inspections", kpiC, "#f97316"),
        ("Estimates", kpiD, "#a855f7"),
        ("Awarded", kpiE, "#22c55e"),
        ("Lost", kpiF, "#ef4444"),
        ("Job Value", f"${kpiG:,.0f}", "#10b981"),
        ("Closed %", f"{kpiE+kpiF}", "#6366f1"),
        ("Won Rate", f"{kpiH:.0f}%", "#8b5cf6"),
        ("Active", f"{kpiB - (kpiE+kpiF)}", "#f59e0b"),
        ("ROI Est", f"{(kpiG/10000):.2f}x", "#ef4444")
    ]

    for i, (title, val, col) in enumerate(metrics):
        with grid[i%4]:
            st.markdown(f"""
<div class='metric-card-dark'>
  <div class='metric-title'>{title}</div>
  <div class='metric-number' style='color:{col}'>{val}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ============ PRIORITY TOP 8 ============
    st.markdown("### 🎯 Priority Leads (Top 8)")
    for _, r in pr_df.head(8).iterrows():
        html = f"""
<div style="background:#111;padding:18px;border-radius:14px;margin-bottom:12px;border:1px solid #333;">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="flex:1;">
      <div style="font-size:18px;font-weight:700;color:white;margin-bottom:6px;">
         #{int(r.get('id'))} — {r.get('contact_name')}
      </div>
      <div style="font-size:13px;color:#aaa;margin-bottom:4px;">
        {r.get('damage_type').title()} | Est: ${r.get('estimated_value'):,.0f}
      </div>
      <div style="font-size:13px;color:red;font-weight:700;">
        ⏳ {int(r.get('time_left_hours'))}h {int((r.get('time_left_hours')*60)%60)}m left
      </div>
    </div>
    <div style="text-align:right; padding-left:20px;">
       <div style="font-size:28px;font-weight:800;color:{r.get('priority_score',0)*'#ef4444'};">
            {r.get('priority_score',0):.2f}
       </div>
       <div style="font-size:11px;color:#6b7280;text-transform:uppercase;">
            Priority
       </div>
    </div>
  </div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("### 📋 All Leads")
    for lead in s.query(Lead).all():
        label = f"#{lead.id} — {lead.contact_name} ({lead.status})"
        with st.expander(label):
            with st.form(f"{lead.id}"):
                status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status))
                estimate_val = st.number_input("Job Value Estimate ($)", value=lead.estimated_value or 0.0)
                invoice = None
                if status == LeadStatus.AWARDED:
                    invoice = st.file_uploader("Upload Invoice File (optional)", key=f"inv_{lead.id}")
                notes = st.text_area("Notes", value=lead.notes or "")
                if st.form_submit_button("Update Lead", type="primary"):
                    lead.status = status
                    lead.estimated_value = estimate_val
                    if invoice:
                        lead.invoice_file = f"invoice_{lead.id}_{invoice.name}"
                    lead.created_at = datetime.utcnow()
                    s.add(lead)
                    s.commit()
                    st.success("Updated ✅")
                    st.rerun()

elif menu == "Analytics":
    st.header("📊 Analytics Dashboard")

    df = leads_df(s)

    fig, ax = plt.subplots()
    ax.pie(df.groupby("status").size(), labels=LeadStatus.ALL)
    centre = plt.Circle((0,0),0.6,fc='white')
    fig.gca().add_artist(centre)
    ax.set_title("Pipeline Stages (Donut)")
    st.pyplot(fig)

    st.markdown("---")

    fig2, ax2 = plt.subplots()
    ax2.pie(df.groupby("qualified").size(), labels=["Unqualified","Qualified"])
    ax2.set_title("Lead Qualification (Pie)")
    st.pyplot(fig2)

    st.markdown("---")
    st.caption("CPA per won job: trending downward MoM, segmented by source")
    st.caption("Velocity: always improving; >48–72 hours stagnation = red flag lead")
