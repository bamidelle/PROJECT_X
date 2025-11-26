import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# ==================== DATABASE SETUP ====================
Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class LeadStatus:
    NEW = "New"
    ACTIVE = "Active"
    QUALIFIED = "Qualified"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, ACTIVE, QUALIFIED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]

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
    status = Column(String, default=LeadStatus.NEW)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    invoice_file = Column(String, nullable=True)
    status_updated_at = Column(DateTime, default=datetime.utcnow)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float, default=0.0)
    sent_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def leads_df(session):
    leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    data = [{
        "id": l.id,
        "contact_name": l.contact_name,
        "contact_phone": l.contact_phone,
        "contact_email": l.contact_email,
        "property_address": l.property_address,
        "damage_type": l.damage_type,
        "assigned_to": l.assigned_to,
        "estimated_value": l.estimated_value or 0,
        "notes": l.notes,
        "status": l.status,
        "sla_hours": l.sla_hours,
        "sla_entered_at": l.sla_entered_at,
        "created_at": l.created_at,
        "qualified": l.qualified,
        "inspection_scheduled": l.inspection_scheduled,
        "inspection_completed": l.inspection_completed,
        "estimate_submitted": l.estimate_submitted,
        "invoice_file": l.invoice_file,
        "status_updated_at": l.status_updated_at
    } for l in leads]
    return pd.DataFrame(data)

def predict_lead_pr(lead_row):
    MODEL_PATH = "lead_conversion_model.pkl"
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        features = [[
            float(lead_row.get("estimated_value") or 0),
            1 if str(lead_row.get("status")).lower() in ["qualified","awarded"] else 0
        ]]
        return round(float(model.predict_proba(features)[0][1]), 3)
    except:
        return None

# ================= UI STYLE =================
st.markdown("""
<style>
body { background-color: #ffffff; font-family: 'Poppins', 'Comfortaa', sans-serif; }
.metric-card {
    background: #111;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    color: white;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    animation: fadeIn 0.6s ease-in-out;
}
@keyframes fadeIn { from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:translateY(0);} }
.metric-value { font-size: 40px; font-weight: 800; }
.metric-title { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
.progress-bar { width:100%; height:8px; border-radius:4px; background:#222; overflow:hidden; margin-top:10px; }
.progress-fill { height:100%; border-radius:4px; width:0%; transition: width 0.5s ease; }
button {
    width:260px; padding:10px 18px; font-size:15px; font-weight:600; border:none; border-radius:10px;
    cursor:pointer; transition:0.3s; animation: btnPop 0.4s ease;
}
@keyframes btnPop {0%{transform:scale(0.95);} 60%{transform:scale(1.03);} 100%{transform:scale(1);}}
button:hover{transform:translateY(-2px); opacity:0.9;}
</style>
""", unsafe_allow_html=True)

# ==================== APP ====================
st.session_state.current_font = "Poppins + Comfortaa Bold"

init_db()
s = get_session()
df = leads_df(s)

# ==================== PIPELINE BOARD ====================
page = st.sidebar.selectbox("Navigation", ["Lead Capture", "Pipeline Board", "Analytics & SLA"])

if page == "Lead Capture":
    st.header("📥 Capture Lead")
    with st.form("new_lead"):
        st.text_input("Name", key="name")
        st.text_input("Phone", key="phone")
        st.text_input("Email", key="email")
        st.number_input("Estimated Value ($)", min_value=0.0, step=100.0, key="value")
        sla = st.number_input("SLA Hours", min_value=1, max_value=168, value=24)
        if st.form_submit_button("Create Lead"):
            add = Lead(
                contact_name=st.session_state.name,
                contact_phone=st.session_state.phone,
                contact_email=st.session_state.email,
                estimated_value=st.session_state.value,
                sla_hours=sla,
                sla_entered_at=datetime.utcnow(),
                status_updated_at=datetime.utcnow(),
                qualified=True,
                status=LeadStatus.NEW
            )
            s.add(add)
            s.commit()
            st.success("Lead Saved ✅")
            st.rerun()

elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Google Ads style")

    if df.empty:
        st.info("No leads yet.")
        st.stop()

    # KPI METRICS
    total = len(df)
    qualified = len(df[df['status'].str.lower().isin(["qualified","awarded"])])
    awarded = len(df[df['status']=="Awarded"])
    lost = len(df[df['status']=="Lost"])
    closed = awarded + lost
    conversion = (awarded/closed*100) if closed else 0
    pipeline_val = df['estimated_value'].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f"<div class='metric-card'><div class='metric-title'>TOTAL</div><div class='metric-value'>{total}</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='metric-card'><div class='metric-title'>QUALIFIED</div><div class='metric-value'>{qualified}</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='metric-card'><div class='metric-title'>WON</div><div class='metric-value'>{awarded}</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='metric-card'><div class='metric-title'>LOST</div><div class='metric-value'>{lost}</div></div>", unsafe_allow_html=True)
    k5.markdown(f"<div class='metric-card'><div class='metric-title'>CONVERSION</div><div class='metric-value'>{conversion:.1f}%</div></div>", unsafe_allow_html=True)
    k6.markdown(f"<div class='metric-card'><div class='metric-title'>PIPELINE VALUE</div><div class='metric-value>${pipeline_val:,.0f}</div></div>", unsafe_allow_html=True)

    # STAGE BREAKDOWN (2 ROWS, 4 COLS)
    stages = df['status'].value_counts().to_dict()
    colors = {
        "New":"#2563eb","Active":"#facc15","Qualified":"#3b82f6","Inspection Scheduled":"#f97316",
        "Inspection Completed":"#14b8a6","Estimate Submitted":"#a855f7","Awarded":"#22c55e","Lost":"#ef4444"
    }
    all_stages = list(colors.keys())

    grid = st.container()
    c = 0
    for row_block in range(2):
        cols = grid.columns(4)
        for i in range(4):
            if c >= len(all_stages): break
            stage = all_stages[c]
            num = stages.get(stage, 0)
            pct = (num/total*100) if total else 0
            col = cols[i]
            with col:
                st.markdown(f"""
                <div class="metric-card" style="background:#111;">
                  <div class="metric-title">{stage.upper()}</div>
                  <div class="metric-value" style="color:{colors[stage]};">{num}</div>
                  <div class="progress-bar"><div class="progress-fill" style="background:{colors[stage]}; width:{pct}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)
            c+=1

    # ALL LEADS EDITABLE
    for _, r in df.head(8).iterrows():
        with st.expander(f"#{int(r['id'])} — {r['contact_name']}"):
            new_status = st.selectbox("Move Stage", LeadStatus.ALL, index=LeadStatus.ALL.index(r['status']))
            invoice=None
            if new_status=="Awarded":
                invoice = st.file_uploader("Upload INVOICE (only for awarded jobs)", type=["pdf","png","jpg"], key=f"invoice_{r['id']}")
            if st.button("SAVE UPDATE", key=f"save_{r['id']}"):
                lead = s.query(Lead).get(int(r['id']))
                lead.status = new_status
                lead.status_updated_at=datetime.utcnow()
                if invoice:
                    path = save_uploaded_file(invoice, lead.id)
                    lead.invoice_file=path
                s.add(lead)
                s.commit()
                st.success("Updated ✅")
                st.rerun()

    # PRIORITY LIST AT BOTTOM
    st.markdown("### 🎯 Priority Leads (Top 8)")
    for _, r in df.iterrows():
        c = {
            "id": r['id'],
            "contact_name": r['contact_name'],
            "estimated_value":r['estimated_value'],
            "status":r['status'],
            "damage_type":r['damage_type'],
            "time_left_hours": r['sla_hours'] - ((datetime.utcnow()-r['created_at']).total_seconds()/3600),
            "priority_score": 1.0 if r['status']=="Awarded" else 0.5
        }
        html = f"""
        <div class="metric-card" style="background:#111;">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <div style="font-size:14px; font-weight:700; color:{priority_color};">{priority_label}</div>
                    <div style="font-size:18px; font-weight:700; color:white;">#{c['id']} — {c['contact_name']}</div>
                    <div style="font-size:13px; color:#aaa;">{c['damage_type']} | Est: <span style="color:#22c55e;">${c['estimated_value']:,.0f}</span></div>
                    <div style="font-size:13px; color:#ef4444;">SLA: {int(c['time_left_hours'])}h {(int(c['time_left_hours']*60)%60)}m</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:28px; font-weight:800; color:#ef4444;">{c['priority_score']:.2f}</div>
                    <div style="font-size:11px; color:#777;">Priority</div>
                </div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

# ==================== ANALYTICS ====================
elif page == "Analytics & SLA":
    st.header("📈 Analytics (Auto-update)")
    if df.empty:
        st.info("No leads for analytics.")
        st.stop()

    funnel = df.groupby("status").size().reset_index(name="Count")
    fig, ax = plt.subplots()
    ax.pie(funnel["Count"])
    st.pyplot(fig)
