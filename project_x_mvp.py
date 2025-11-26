# =================== BEGIN FULL APP CODE ===================

import streamlit as st
from datetime import datetime, timedelta
import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------- Database Setup ----------
Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}")
SessionLocal = sessionmaker(bind=engine)

# ---------- Lead Status ----------
class LeadStatus:
    NEW = "New"
    ACTIVE = "Active"
    QUALIFIED = "Qualified"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"

    ALL = [NEW, ACTIVE, QUALIFIED, ESTIMATE_SUBMITTED, AWARDED, LOST]

# ---------- Database Models ----------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    assigned_to = Column(String)
    estimated_value = Column(Float, default=0.0)
    notes = Column(String)
    status = Column(String, default=LeadStatus.NEW)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    invoice_path = Column(String, nullable=True)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float, default=0.0)
    sent_at = Column(DateTime, nullable=True)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ---------- DB Init ----------
def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def leads_df(session):
    leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    return pd.DataFrame([
        {
            "id": l.id,
            "contact_name": l.contact_name,
            "estimated_value": l.estimated_value or 0,
            "status": l.status,
            "qualified": l.qualified,
            "created_at": l.created_at,
            "sla_entered_at": l.sla_entered_at,
            "sla_hours": l.sla_hours,
            "damage_type": l.damage_type,
            "contact_phone": l.contact_phone,
            "contact_email": l.contact_email,
            "assigned_to": l.assigned_to,
            "notes": l.notes,
            "invoice_path": l.invoice_path
        } for l in leads
    ])

def estimates_df(session):
    est = session.query(Estimate).all()
    return pd.DataFrame([
        {
            "id": e.id,
            "lead_id": e.lead_id,
            "amount": e.amount,
            "approved": e.approved,
            "lost": e.lost,
            "lost_reason": e.lost_reason,
            "created_at": e.created_at
        } for e in est
    ])

# ---------- Core Helpers ----------
def save_uploaded_file(uploaded_file, lead_id, folder="uploaded_invoices"):
    if not uploaded_file: return None
    os.makedirs(folder, exist_ok=True)
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    open(path, "wb").write(uploaded_file.getbuffer())
    return path

def create_estimate(session, lead_id, amount, details=None):
    e = Estimate(lead_id=lead_id, amount=float(amount))
    session.add(e); session.commit()
    return e

def mark_estimate_sent(session, est_id):
    e = session.get(Estimate, est_id)
    if e: e.sent_at = datetime.utcnow(); session.commit()

def mark_estimate_approved(session, est_id):
    e = session.get(Estimate, est_id)
    if e:
        e.approved = True
        l = session.get(Lead, e.lead_id)
        if l: l.status = LeadStatus.AWARDED
        session.commit()

def mark_estimate_lost(session, est_id, reason=None):
    e = session.get(Estimate, est_id)
    if e:
        e.lost = True; e.lost_reason = reason
        l = session.get(Lead, e.lead_id)
        if l: l.status = LeadStatus.LOST
        session.commit()

def remaining_sla_hms(sla_entered_at, sla_hours):
    if not sla_entered_at: sla_entered_at = datetime.utcnow()
    if isinstance(sla_entered_at, str):
        try: sla_entered_at = datetime.fromisoformat(sla_entered_at)
        except: sla_entered_at = datetime.utcnow()
    if pd.isna(sla_entered_at): sla_entered_at = datetime.utcnow()

    deadline = sla_entered_at + timedelta(hours=int(sla_hours or 24))
    remaining = deadline - datetime.utcnow()

    if remaining.total_seconds() <= 0:
        return "<span style='color:#ef4444;font-weight:700;'>00:00:00 ❗ OVERDUE</span>"
    else:
        h = int(remaining.total_seconds() // 3600)
        m = int((remaining.total_seconds() % 3600) // 60)
        return f"<span style='color:#ef4444;font-weight:600;'>{h}h {m}m left</span>"

# ---------- UI Styles ----------
st.set_page_config(page_title="Project X Pipeline", layout="wide")

st.markdown("""
<style>

body, .stApp { background-white; }

.metric-card {
  background:#111;
  padding:18px;
  border-radius:16px;
  border:1px solid #333;
  margin:10px 0;
  color:white;
  animation:fadeIn 0.4s ease;
}

.metric-label { font-size:14px; color:white; font-weight:600; }

.metric-value { font-size:36px; font-weight:800; }

.stage-badge {
  padding:4px 10px;
  border-radius:12px;
  font-size:12px;
  font-weight:600;
  margin-left:6px;
}

.progress-bar {
  width:100%;
  height:6px;
  background:#222;
  border-radius:3px;
  overflow:hidden;
  margin-top:10px;
}

.progress-fill {
  height:100%;
  border-radius:3px;
  transition:width 0.5s ease;
}

@keyframes fadeIn { 0% {opacity:0; transform:translateY(5px)} 100% {opacity:1; transform:translateY(0)} }

button[data-testid="baseButton-secondary"], .stFormSubmitButton button {
  padding:10px 24px;
  border-radius:10px;
  font-size:15px;
  font-weight:600;
  min-width:160px;
  transition:all 0.3s ease;
}

.stFormSubmitButton button:hover { transform:scale(1.04); }

</style>
""", unsafe_allow_html=True)

init_db()

# ---------- Navigation ----------
page = st.sidebar.selectbox("Navigate", ["Lead Capture","Pipeline Board","Analytics"])

s = get_session()

# ---------- Lead Capture ----------
if page == "Lead Capture":
    st.header("📥 Lead Capture")
    with st.form("lead_form"):
        contact_name = st.text_input("Full Name*")
        contact_phone = st.text_input("Phone*")
        contact_email = st.text_input("Email")
        property_address = st.text_input("Property Address*")
        damage_type = st.selectbox("Damage Type", ["Water", "Fire", "Mold", "Reconstruction"])
        assigned_to = st.text_input("Assigned To")
        sla_hours = st.number_input("SLA response hours", min_value=1,value=24)

        st.markdown("💰 **Job Value Estimate (US Dollars)**")
        estimated_value = st.number_input("Estimated Job Value ($)", min_value=0.0, step=100.0)

        notes = st.text_area("Notes")

        if st.form_submit_button("➕ Save Lead"):
            new_lead = Lead(
                contact_name=contact_name,
                contact_phone=contact_phone,
                contact_email=contact_email,
                property_address=property_address,
                damage_type=damage_type,
                assigned_to=assigned_to,
                estimated_value=estimated_value,
                notes=notes,
                sla_hours=sla_hours,
                sla_entered_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                qualified=True
            )
            s.add(new_lead)
            s.commit()
            st.success("Lead saved ✅")

# ---------- Pipeline Board (2 Rows, 4 Cols) ----------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard")

    df = leads_df(s)
    total_leads = len(df)

    # Compute funnel
    funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
    funnel.columns = ["status", "count"]
    funnel["percent"] = (funnel["count"]/total_leads*100).round(1)

    stage_colors = {
        "New":"#2563eb",
        "Active":"#f59e0b",
        "Qualified":"#14b8a6",
        "Estimate Submitted":"#a855f7",
        "Awarded":"#22c55e",
        "Lost":"#ef4444"
    }

    # Build cards
    cards = []
    for _,r in funnel.iterrows():
        color = stage_colors.get(r["status"],"#888")
        cards.append({"status":r["status"],"count":r["count"],"percent":r["percent"],"color":color})

    # 2×4 grid
    grid = st.columns(4)
    row = 0
    for i,card in enumerate(cards):
        if i>3 and row==0:
            st.markdown("<br>",unsafe_allow_html=True)
            grid = st.columns(4)
            row = 1

        with grid[i % 4]:
            st.markdown(f"""
<div class="metric-card">
  <div style="color:{card['color']}; font-size:13px; font-weight:700; margin-bottom:6px;">{card['status']}</div>
  <div class="metric-value" style="color:{card['color']};">{card['count']}</div>
  <div class="progress-bar"><div class="progress-fill" style="background:{card['color']}; width:{card['percent']}%;"></div></div>
  <div style="margin-top:6px; font-size:12px; color:white; font-weight:600;">{card['percent']}%</div>
</div>
""", unsafe_allow_html=True)

    # -------- Priority Leads --------
    priority_list = []
    for _,r in df.iterrows():
        priority_list.append({
            "id":r["id"],
            "contact_name":r["contact_name"],
            "status":r["status"],
            "estimated_value":r["estimated_value"],
            "damage_type":r["damage_type"],
            "sla_hms": remaining_sla_hms(r.get("sla_entered_at"), r.get("sla_hours"))
        })

    pr_df = pd.DataFrame(priority_list)

    st.markdown("### 🎯 Priority Leads (Top 8)")
    for _,r in pr_df.head(8).iterrows():
        html = f"""
<div class="metric-card" style="background:#222;border:1px solid #ccc;">
  <div class="stage-badge" style="background:{stage_colors.get(r['status'],'#777')}20;color:white;">{r['status']}</div>
  <div style="font-size:18px;font-weight:700;color:white;margin-top:6px;">#{r['id']} — {r['contact_name']}</div>
  <div style="font-size:13px;color:#eee;margin-top:4px;">{r['damage_type']} | Est: ${r['estimated_value']:,.0f}</div>
  <div style="margin-top:6px;font-size:13px;">{r['sla_hms']}</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    # -------- Editable Lead Progression --------
    st.markdown("### 📋 All Leads")

    for lead in leads:
        title = f"#{lead.id} — {lead.contact_name} — {lead.damage_type} — ${lead.estimated_value:,.0f}"

        with st.expander(title):
            new_status = st.selectbox("Move Stage", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status))
            invoice = None
            if new_status == LeadStatus.AWARDED:
                invoice = st.file_uploader("Upload Invoice (optional)", key=f"inv_{lead.id}")

            if st.button("Update Stage + Analytics"):
                lead.status = new_status
                if invoice:
                    lead.invoice_path = save_uploaded_file(invoice, lead.id)
                s.add(lead); s.commit()
                st.rerun()

# ---------- Analytics ----------
elif page == "Analytics":
    st.header("📊 Analytics (Auto Updates)")
    df = leads_df(s)
    pie = df["status"].value_counts()

    plt.figure()
    plt.pie(pie.values, labels=pie.index)
    plt.title("Pipeline Conversion Funnel")
    st.pyplot(plt)

# =================== END FULL APP CODE ===================
