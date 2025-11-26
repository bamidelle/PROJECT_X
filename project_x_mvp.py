import streamlit as st
import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

# =========================================================
# DATABASE SETUP
# =========================================================

Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# =========================================================
# ENUM FOR LEAD STATUS
# =========================================================

class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]

# =========================================================
# DATABASE MODELS
# =========================================================

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

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float, default=0.0)
    sent_at = Column(DateTime)
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
        "estimated_value": l.estimated_value,
        "status": l.status,
        "sla_hours": l.sla_hours,
        "sla_entered_at": l.sla_entered_at,
        "created_at": l.created_at,
        "qualified": l.qualified,
        "damage_type": l.damage_type
    } for l in leads]
    return pd.DataFrame(data)

def add_lead(session, **kwargs):
    lead = Lead(**kwargs)
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead

def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount, sent_at=datetime.utcnow())
    session.add(est)
    session.commit()
    return est

# =========================================================
# STREAMLIT UI SETUP
# =========================================================

st.set_page_config(page_title="Project X", layout="wide")
init_db()

if "weights" not in st.session_state:
    st.session_state.weights = {
        "value_baseline": 6000.0,
        "value_weight": 0.5,
        "sla_weight": 0.35,
        "urgency_weight": 0.15,
    }

# ==========================================================
# KPI CARDS CALCULATION
# ==========================================================

def compute_funnel_metrics(df):
    funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).to_dict()
    return funnel

def render_kpi_card(title, value, color):
    st.markdown(f"""
    <div style="width:100%; background:{color}; padding:18px; border-radius:16px; margin-bottom:12px; box-shadow:0 4px 8px rgba(0,0,0,0.15);">
      <div style="font-size:14px; font-weight:600; color:white; text-transform:uppercase;">{title}</div>
      <div style="font-size:36px; font-weight:800; color:white; margin-top:6px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_lead_stage_card(stage, count, color):
    percent = (count / max(st.session_state.total_leads, 1)) * 100
    st.markdown(f"""
    <div class="metric-card" style="background:#111; padding:18px; border-radius:16px; margin-bottom:12px; animation:fadeIn 0.6s;">
        <div style="font-size:15px; font-weight:600; color:white;">{stage.upper()}</div>
        <div style="font-size:36px; font-weight:800; color:{color}; margin-top:6px;">{count}</div>
        <div class="progress-bar" style="margin-top:12px;">
          <div class="progress-fill" style="background:{color}; width:{percent}%; height:8px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================================
# PRIORITY CARD (FIXED, NO BROKEN TAGS)
# ==========================================================

def render_priority_lead_card(r, stage_colors):
    score = r.get("priority_score", 0)
    status_color = stage_colors.get(r.get("status"), "#000000")

    if score >= 0.7:
        priority_color = "#ef4444"
        priority_label = "🔴 CRITICAL"
    elif score >= 0.45:
        priority_color = "#f97316"
        priority_label = "🟠 HIGH"
    else:
        priority_color = "#22c55e"
        priority_label = "🟢 NORMAL"

    remaining_hours = r.get("time_left_hours", 0)
    if remaining_hours <= 0:
        sla_html = "<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
    else:
        h = int(remaining_hours)
        m = int((remaining_hours * 60) % 60)
        sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {h}h {m}m left</span>"

    html = f"""
    <div style="background:#111; padding:18px; border-radius:16px; margin-bottom:12px; border:1px solid #333; animation:fadeIn 0.6s;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="flex:1;">
          <div style="margin-bottom:8px;">
            <span style="color:{priority_color}; font-weight:700; font-size:14px;">{priority_label}</span>
            <span style="background:{status_color}20; color:{status_color}; border:1px solid {status_color}40; padding:4px 10px; border-radius:14px; font-size:11px; font-weight:600; margin-left:8px;">
              {r.get('status')}
            </span>
          </div>
          <div style="font-size:20px; font-weight:700; color:white;">#{int(r.get('id'))} — {r.get('contact_name')}</div>
          <div style="font-size:13px; color:#aaa; margin-top:4px;">
            {r.get('damage_type').title()} | Est: <span style="color:#22c55e; font-weight:700;">${r.get('estimated_value'):,.0f}</span>
          </div>
          <div style="margin-top:6px; font-size:13px;">{sla_html}</div>
        </div>
        <div style="text-align:right; padding-left:20px;">
          <div style="font-size:28px; font-weight:800; color:{priority_color};">{score:.2f}</div>
          <div style="font-size:11px; color:#777; text-transform:uppercase;">Priority</div>
        </div>
      </div>
    </div>
    """
    return st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# GLASSY PIPELINE UI
# ==========================================================

st.markdown("""
<style>
body { background:white; font-family:Poppins, Comfortaa; }
@keyframes fadeIn { from{opacity:0} to{opacity:1} }
.metric-card { transition:0.3s; }
.metric-card:hover { opacity:0.85; transform:translateY(-3px); }
button {
    width:100%;
    padding:10px;
    font-size:15px;
    font-weight:700;
    border:none;
    border-radius:10px;
    transition:0.3s;
}
button:hover {
    transform:scale(1.03);
}
.progress-bar { background:#ddd; }
.progress-fill { height:8px; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# MAIN PIPELINE SECTION (2 rows × 4 columns)
# ==========================================================

page = st.sidebar.selectbox("Navigate", ["Lead Capture", "Pipeline Board", "Analytics"])

s = get_session()
df = leads_df(s)
st.session_state.total_leads = len(df)

if page == "Pipeline Board":

    st.markdown("### 📊 Key Performance Indicators")

    st.session_state.funnel = compute_funnel_metrics(df)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)

    closed = st.session_state.funnel["Awarded"] + st.session_state.funnel["Lost"]

    st.session_state.kpis = {
        1: ("Lead Compliance", len(df[df.sla_entered_at <= df.created_at + pd.to_timedelta(df.sla_hours, "h")]), "#2563eb"),
        2: ("Qualification Rate", st.session_state.funnel["New"], "#eab308"),
        3: ("Inspection Booked", st.session_state.funnel["Inspection Scheduled"], "#f97316"),
        4: ("Estimate Won Rate", st.session_state.funnel["Awarded"], "#22c55e"),
        5: ("CPA Efficiency", st.session_state.funnel["Estimate Submitted"], "#14b8a6"),
        6: ("Velocity KPI", st.session_state.funnel["Inspection Completed"], "#a855f7"),
        7: ("Lost Leads", st.session_state.funnel["Lost"], "#ef4444"),
        8: ("Active Leads", st.session_state.total_leads - closed, "#4ade80"),
    }

    for i, (title, value, color) in st.session_state.kpis.items():
        if i <= 4:
            with [kpi1, kpi2, kpi3, kpi4][i-1]:
                render_kpi_card(title, value, color)
        else:
            with [kpi5, kpi6, kpi7, kpi8][i-5]:
                render_kpi_card(title, value, color)

    st.markdown("---")

    stage1, stage2, stage3, stage4, stage5, stage6, stage7 = st.columns(7)

    for i, stage in enumerate(LeadStatus.ALL):
        with [stage1, stage2, stage3, stage4, stage5, stage6, stage7][i]:
            render_lead_stage_card(stage, st.session_state.funnel.get(stage, 0), stage_colors=st.session_state.stage_colors)

    st.markdown("---")

    st.markdown("### 🎯 Priority Leads (Top 8)")
    priority_list = []

    for _, row in df.iterrows():
        priority_list.append({
            "id": row.id,
            "contact_name": row.contact_name,
            "estimated_value": row.estimated_value,
            "status": row.status,
            "priority_score": min(row.estimated_value/6000, 1.0),
            "time_left_hours": max((datetime.utcnow() + timedelta(hours=row.sla_hours) - datetime.utcnow()).total_seconds()/3600, 0),
            "damage_type": row.damage_type
        })

    pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

    for _, r in pr_df.head(8).iterrows():
        render_priority_lead_card(r, st.session_state.stage_colors)

# ==========================================================
# LEAD CAPTURE AND EDITING SECTION (Editable stages)
# ==========================================================

if page == "Lead Capture":
    st.markdown("## ➕ New Lead")
    with st.form("lead_form"):
        st.text_input("Full Name", key="contact_name")
        st.text_input("Phone", key="contact_phone")
        st.text_input("Email", key="contact_email")
        st.text_input("Property Address", key="property_address")
        st.selectbox("Damage Type", ["Water", "Fire", "Mold", "Storm", "Reconstruction"], key="damage_type")
        st.text_area("Notes", key="notes")
        st.number_input("SLA Hours", min_value=1, max_value=168, value=24, step=1, key="sla_hours")
        st.number_input("Job Value Estimate (USD)", min_value=0.0, step=100.0, key="estimated_value")

        if st.form_submit_button("Submit Lead"):
            new = add_lead(s,
                contact_name=st.session_state.contact_name or None,
                contact_phone=st.session_state.contact_phone or None,
                contact_email=st.session_state.contact_email or None,
                property_address=st.session_state.property_address or None,
                damage_type=st.session_state.damage_type.lower(),
                assigned_to=None,
                estimated_value=st.session_state.estimated_value or 0.0,
                notes=st.session_state.notes or None,
                sla_hours=st.session_state.sla_hours or 24,
                sla_entered_at=datetime.utcnow(),
                qualified=True,
                status=LeadStatus.NEW
            )
            st.success(f"Lead #{new.id} captured!")
            st.rerun()

# ==========================================================
# ANALYTICS SECTION (Pie chart auto update)
# ==========================================================

elif page == "Analytics":
    st.markdown("## 📈 Lead Stage Distribution")
    pie_data = df.groupby("status").size().reset_index(name="count")

    import matplotlib.pyplot as plt
    plt.pie(pie_data["count"], labels=pie_data["status"])
    st.pyplot(plt)

# ==========================================================
# END OF CODE
# ==========================================================
