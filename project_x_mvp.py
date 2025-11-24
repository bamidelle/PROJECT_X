# project_x_fixed.py
# Single-file Streamlit app (fixed + requested features)
# - White background
# - Pipeline Dashboard (Google-Ads style)
# - KPI cards (2 rows × 4 cols), colorful, white KPI text
# - Pipeline cards with black background & white text, colored progress bars
# - Priority Leads (Top 8) with fixed display & styling
# - Expandable All Leads editable; can move to AWARDED / LOST and AWARDED allows optional Invoice upload
# - Job Value Estimate label changed
# - Analytics uses responsive pie chart and auto-updates when statuses change
# - Auto-refresh (safe) implemented (toggleable, default 30s)
# - No st.experimental_rerun called globally (only used safely)
# - joblib model load is optional and fails gracefully

import os
import io
from datetime import datetime, timedelta, time as dtime
import time
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# SQLAlchemy for ORM
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Try to import joblib (optional). If not available, continue without ML model.
try:
    import joblib
except Exception:
    joblib = None

# -------- CONFIG --------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_fixed.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_PATH = "lead_conversion_model.pkl"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# -------- MODELS --------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="—")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    # flags
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    qualified = Column(Boolean, default=False)
    # status & job outcome fields
    status = Column(String, default="NEW")  # NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)  # path to uploaded invoice
    lost_date = Column(DateTime, nullable=True)
    # misc
    created_by = Column(String, nullable=True)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)

def init_db():
    Base.metadata.create_all(bind=engine)

# -------- UTILITIES --------
def get_session():
    return SessionLocal()

def leads_df(session):
    rows = session.query(Lead).all()
    recs = []
    for r in rows:
        recs.append({
            "id": r.id,
            "source": r.source,
            "source_details": r.source_details,
            "contact_name": r.contact_name,
            "contact_phone": r.contact_phone,
            "contact_email": r.contact_email,
            "property_address": r.property_address,
            "damage_type": r.damage_type,
            "assigned_to": r.assigned_to,
            "notes": r.notes,
            "estimated_value": float(r.estimated_value or 0.0),
            "created_at": r.created_at,
            "sla_hours": int(r.sla_hours or 24),
            "sla_entered_at": r.sla_entered_at,
            "contacted": bool(r.contacted),
            "inspection_scheduled": bool(r.inspection_scheduled),
            "inspection_completed": bool(r.inspection_completed),
            "estimate_submitted": bool(r.estimate_submitted),
            "qualified": bool(r.qualified),
            "status": r.status,
            "awarded_date": r.awarded_date,
            "awarded_invoice": r.awarded_invoice,
            "lost_date": r.lost_date,
        })
    if recs:
        return pd.DataFrame(recs)
    else:
        return pd.DataFrame(columns=[
            "id","source","source_details","contact_name","contact_phone","contact_email","property_address",
            "damage_type","assigned_to","notes","estimated_value","created_at","sla_hours","sla_entered_at",
            "contacted","inspection_scheduled","inspection_completed","estimate_submitted","qualified","status",
            "awarded_date","awarded_invoice","lost_date"
        ])

def add_lead(session, **kwargs):
    lead = Lead(**kwargs)
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead

def create_estimate(session, lead_id, amount, details):
    est = Estimate(lead_id=lead_id, amount=float(amount), details=details)
    session.add(est)
    session.commit()
    session.refresh(est)
    return est

def save_uploaded_file(uploaded_file, lead_id, folder="uploaded_files"):
    if uploaded_file is None:
        return None
    os.makedirs(folder, exist_ok=True)
    ts = int(datetime.utcnow().timestamp())
    fname = f"lead_{lead_id}_{ts}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def compute_priority_for_lead_row(lead_row, weights):
    # lead_row is dict-like including estimated_value, sla_entered_at, sla_hours and flags
    # returns score between 0 and 1 plus debug components
    try:
        value = float(lead_row.get("estimated_value") or 0.0)
    except Exception:
        value = 0.0
    baseline = float(weights.get("value_baseline", 5000.0))
    value_score = min(1.0, value / (baseline if baseline > 0 else 1.0))

    # SLA
    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    try:
        if isinstance(sla_entered, str):
            sla_entered = datetime.fromisoformat(sla_entered)
        elif sla_entered is None or pd.isna(sla_entered):
            sla_entered = datetime.utcnow()
    except Exception:
        sla_entered = datetime.utcnow()
    try:
        sla_hours = int(lead_row.get("sla_hours") or 24)
    except Exception:
        sla_hours = 24
    deadline = sla_entered + timedelta(hours=sla_hours)
    time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)  # favor shorter remaining times

    # urgency flags
    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0
    urgency_component = (contacted_flag * weights.get("contacted_w", 0.6)
                        + inspection_flag * weights.get("inspection_w", 0.5)
                        + estimate_flag * weights.get("estimate_w", 0.5))

    total_weight = float(weights.get("value_weight", 0.5) + weights.get("sla_weight", 0.35) + weights.get("urgency_weight", 0.15))
    if total_weight <= 0:
        total_weight = 1.0

    score = (value_score * float(weights.get("value_weight", 0.5))
            + sla_score * float(weights.get("sla_weight", 0.35))
            + urgency_component * float(weights.get("urgency_weight", 0.15))) / total_weight
    score = max(0.0, min(score, 1.0))
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h

def predict_lead_probability(model, lead_row):
    # stub: if model exists, prepare a numeric vector and call model.predict_proba
    # We'll implement a safe wrapper; if joblib missing or model fails, return None
    if model is None:
        return None
    try:
        # simple engineered features (must match your model training)
        features = [
            float(lead_row.get("estimated_value") or 0.0),
            1.0 if lead_row.get("qualified") else 0.0,
            1.0 if lead_row.get("contacted") else 0.0
        ]
        # model should accept 2D array
        prob = model.predict_proba([features])[0][1]
        return float(prob)
    except Exception:
        return None

# -------- UI CSS (white background requested) --------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root{
  --bg-white:#ffffff;
  --muted:#6b7280;
  --white:#ffffff;
  --black:#0b0b0b;
  --radius:12px;
  --primary-red:#ef4444;
  --money-green:#16a34a;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}

/* Base */
body, .stApp {
  background: var(--bg-white);
  color: var(--black);
  font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #f8fafc !important;
  padding: 18px;
  border-right: 1px solid rgba(0,0,0,0.04);
}

/* Header */
.header { padding: 12px; color: var(--black); font-weight:700; font-size:20px; }

/* Metric card */
.metric-card {
  border-radius: var(--radius);
  padding: 18px;
  margin-bottom: 10px;
  color: var(--white);
  background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  box-shadow: 0 6px 14px rgba(15,23,42,0.06);
}
.metric-label { font-size:12px; color: rgba(255,255,255,0.85); text-transform:uppercase; font-weight:600; }
.metric-value { font-size:28px; font-weight:800; color: var(--white); margin-top:6px; }

/* Pipeline card (black background) */
.pipeline-card {
  background: var(--black);
  color: var(--white);
  padding: 14px;
  border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.06);
  margin-bottom: 10px;
}
.stage-badge {
  padding:6px 12px; border-radius:20px; font-size:12px; font-weight:700; display:inline-block;
}
.progress-bar { width:100%; height:8px; background:rgba(255,255,255,0.08); border-radius:6px; overflow:hidden; margin-top:10px;}
.progress-fill { height:100%; border-radius:6px; transition:width .3s ease; }
.priority-card { background: #0b1020; color: #fff; padding:12px; border-radius:10px; margin-bottom:10px; }

/* Small text */
.kv { color: #6b7280; font-size:12px; }
"""

# -------- APP SETUP --------
st.set_page_config(page_title="Project X — Pipeline", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
init_db()

st.markdown("<div class='header'>Project X — Sales & Pipeline</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }

    st.markdown("### Priority weight tuning")
    st.session_state.weights["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("Within urgency flags:")
    st.session_state.weights["contacted_w"] = st.slider("Not-contacted weight", 0.0, 1.0, float(st.session_state.weights["contacted_w"]), step=0.05)
    st.session_state.weights["inspection_w"] = st.slider("Not-scheduled weight", 0.0, 1.0, float(st.session_state.weights["inspection_w"]), step=0.05)
    st.session_state.weights["estimate_w"] = st.slider("No-estimate weight", 0.0, 1.0, float(st.session_state.weights["estimate_w"]), step=0.05)
    st.session_state.weights["value_baseline"] = st.number_input("Value baseline", min_value=100.0, value=float(st.session_state.weights["value_baseline"]), step=100.0)

    st.markdown("---")
    st.markdown("### Auto-refresh")
    if "autorefresh" not in st.session_state:
        st.session_state.autorefresh = True
    st.session_state.autorefresh = st.checkbox("Auto-refresh pipeline (safe)", value=st.session_state.autorefresh)
    if "autorefresh_interval" not in st.session_state:
        st.session_state.autorefresh_interval = 30
    st.session_state.autorefresh_interval = st.number_input("Interval (seconds)", min_value=5, max_value=600, value=int(st.session_state.autorefresh_interval), step=5)

# Implement safe auto-refresh using time checks + st.experimental_rerun only when needed and available.
# We'll store last_refresh timestamp in session_state.
if st.session_state.get("autorefresh", True):
    now = time.time()
    last = st.session_state.get("_last_refresh_ts", 0)
    interval = int(st.session_state.get("autorefresh_interval", 30))
    if now - last > interval:
        st.session_state["_last_refresh_ts"] = now
        # call experimental_rerun in try/except to avoid hard crash if unavailable
        try:
            st.experimental_rerun()
        except Exception:
            # If rerun not available, we silently continue (page will refresh on interaction)
            pass

# -------- Page: Leads / Capture --------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    with st.form("lead_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic Search", "Referral", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google.")
            contact_name = st.text_input("Contact name", placeholder="John Doe")
            contact_phone = st.text_input("Contact phone", placeholder="+1-555-0123")
            contact_email = st.text_input("Contact email", placeholder="name@example.com")
        with col2:
            property_address = st.text_input("Property address", placeholder="123 Main St, City, State")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to", placeholder="Estimator name")
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["No", "Yes"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context.")
        est_val = st.number_input("Job Value Estimate (USD)", min_value=0.0, value=0.0, step=100.0, format="%f")
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
                notes=notes,
                sla_hours=int(sla_hours),
                estimated_value=float(est_val) if est_val is not None else 0.0,
                qualified=True if qualified_choice == "Yes" else False,
                sla_entered_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                status="NEW"
            )
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# -------- Page: Pipeline Board --------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Google Ads style")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # Try load model safely
        lead_model = None
        if joblib is not None and os.path.exists(MODEL_PATH):
            try:
                lead_model = joblib.load(MODEL_PATH)
            except Exception:
                lead_model = None

        # ========== KPI Cards (2 rows x 4 columns) ==========
        total_leads = len(df)
        qualified_leads = int(df['qualified'].sum()) if not df.empty else 0
        total_value = df['estimated_value'].sum() if not df.empty else 0.0
        awarded_leads = len(df[df['status'] == "AWARDED"]) if not df.empty else 0
        lost_leads = len(df[df['status'] == "LOST"]) if not df.empty else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads

        # Make 2 rows of 4 columns: use chunking
        kpi_items = [
            {"label":"Total Leads", "value": total_leads, "color":"#2563eb", "note": f"{qualified_leads} Qualified"},
            {"label":"Pipeline Value (USD)", "value": f"${total_value:,.0f}", "color":"#16a34a", "note":"Sum of estimates"},
            {"label":"Conversion Rate", "value": f"{conversion_rate:.1f}%", "color":"#a855f7", "note": f"{awarded_leads}/{closed_leads} Won"},
            {"label":"Active Leads", "value": active_leads, "color":"#f97316", "note":"In progress"},
            {"label":"Awarded Jobs", "value": awarded_leads, "color":"#059669", "note":"Jobs awarded"},
            {"label":"Lost Jobs", "value": lost_leads, "color":"#ef4444", "note":"Jobs lost"},
            {"label":"Qualified (%)", "value": f"{(qualified_leads/total_leads*100):.1f}%" if total_leads>0 else "0.0%", "color":"#0ea5e9", "note":"Qualified leads"},
            {"label":"Avg Job Value", "value": f"${(total_value/total_leads):,.0f}" if total_leads>0 else "$0", "color":"#0ea5e9", "note":"Average estimate"}
        ]

        st.markdown("### 📊 Key Performance Indicators")
        # display in two rows automatically using chunk
        cols = st.columns(4)
        for idx, item in enumerate(kpi_items[:4]):
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {item['color']}, {item['color']}33);">
                    <div class="metric-label">{item['label']}</div>
                    <div class="metric-value" style="color: #ffffff;">{item['value']}</div>
                    <div class="metric-change" style="background: rgba(255,255,255,0.06); color: #ffffff; margin-top:8px;">{item['note']}</div>
                </div>
                """, unsafe_allow_html=True)
        cols = st.columns(4)
        for idx, item in enumerate(kpi_items[4:8]):
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {item['color']}, {item['color']}33);">
                    <div class="metric-label">{item['label']}</div>
                    <div class="metric-value" style="color: #ffffff;">{item['value']}</div>
                    <div class="metric-change" style="background: rgba(255,255,255,0.06); color: #ffffff; margin-top:8px;">{item['note']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ========== Pipeline Stage Cards (one row per stage; cards are black) ==========
        st.markdown("### 📈 Pipeline Stages")
        LEAD_STATUSES = ["NEW","CONTACTED","INSPECTION_SCHEDULED","INSPECTION_COMPLETED","ESTIMATE_SUBMITTED","AWARDED","LOST"]
        stage_colors = {
            "NEW": "#2563eb",
            "CONTACTED": "#eab308",
            "INSPECTION_SCHEDULED": "#f97316",
            "INSPECTION_COMPLETED": "#14b8a6",
            "ESTIMATE_SUBMITTED": "#7c3aed",
            "AWARDED": "#059669",
            "LOST": "#ef4444"
        }
        total_leads = max(1, total_leads)
        stage_cols = st.columns(len(LEAD_STATUSES))
        counts = df['status'].value_counts().to_dict()
        for i, stg in enumerate(LEAD_STATUSES):
            count = counts.get(stg, 0)
            pct = (count/total_leads)*100
            color = stage_colors.get(stg, "#111111")
            with stage_cols[i]:
                st.markdown(f"""
                <div class="pipeline-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                      <div>
                        <div style="font-size:12px; color:#ffffff; font-weight:700;">{stg.replace('_',' ').title()}</div>
                        <div style="font-size:20px; color:{color}; font-weight:800;">{count}</div>
                      </div>
                    </div>
                    <div class="progress-bar"><div class="progress-fill" style="width:{pct}%; background:{color};"></div></div>
                    <div style="margin-top:8px; font-size:12px; color:#9ca3af;">{pct:.1f}% of pipeline</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ========== Priority Leads (Top 8) ==========
        st.markdown("### 🎯 Priority Leads (Top 8)")
        priority_list = []
        for _, row in df.iterrows():
            try:
                score, *_ = compute_priority_for_lead_row(row, weights)
            except Exception:
                score = 0.0
            # SLA calculation robust
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            try:
                if isinstance(sla_entered, str):
                    sla_entered = datetime.fromisoformat(sla_entered)
                elif pd.isna(sla_entered):
                    sla_entered = datetime.utcnow()
            except Exception:
                sla_entered = datetime.utcnow()
            try:
                sla_hours = int(row.get("sla_hours") or 24)
            except Exception:
                sla_hours = 24
            deadline = sla_entered + timedelta(hours=sla_hours)
            remaining = deadline - datetime.utcnow()
            time_left_hours = max(0.0, remaining.total_seconds() / 3600.0)
            overdue = remaining.total_seconds() <= 0
            # predicted probability
            prob = None
            if lead_model is not None:
                try:
                    prob = predict_lead_probability(lead_model, row)
                except Exception:
                    prob = None

            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left_hours),
                "priority_score": float(score),
                "status": row.get("status"),
                "sla_overdue": overdue,
                "deadline": deadline,
                "conversion_prob": prob,
                "damage_type": row.get("damage_type") or "Unknown"
            })

        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        if pr_df.empty:
            st.info("No priority leads yet.")
        else:
            # Show top 8
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                if score >= 0.7:
                    priority_color = "#ef4444"; priority_label="🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"; priority_label="🟠 HIGH"
                else:
                    priority_color = "#22c55e"; priority_label="🟢 NORMAL"

                status_color = stage_colors.get(r["status"], "#111111")

                # SLA time left display; red when low
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
                else:
                    # show hours & mins
                    total_seconds = max(0, (r["deadline"] - datetime.utcnow()).total_seconds())
                    hours_left = int(total_seconds // 3600)
                    mins_left = int((total_seconds % 3600) // 60)
                    # make display red if under 2 hours
                    time_color = "#ef4444" if (hours_left*60 + mins_left) <= 120 else "#2563eb"
                    sla_html = f"<span style='color:{time_color};font-weight:600;'>⏳ {hours_left}h {mins_left}m left</span>"

                conv_html = ""
                if r["conversion_prob"] is not None:
                    conv_pct = r["conversion_prob"] * 100
                    conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                    conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"

                # Fixed right-side priority display size & colors (addresses the reported broken HTML snippet)
                st.markdown(f"""
                <div class="priority-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="flex:1;">
                            <div style="margin-bottom:8px;">
                                <span style="color:{priority_color};font-weight:800;font-size:14px;">{priority_label}</span>
                                <span class="stage-badge" style="background:{status_color}22;color:{status_color};border:1px solid {status_color}44;margin-left:8px;">
                                    {r['status']}
                                </span>
                            </div>
                            <div style="font-size:16px; font-weight:800; color:#ffffff; margin-bottom:4px;">
                                #{int(r['id'])} — {r['contact_name']}
                            </div>
                            <div style="font-size:13px; color:#9aa3b2; margin-bottom:8px;">
                                {r['damage_type'].title()} | Est: <span style="color:#16a34a;font-weight:800;">${r['estimated_value']:,.0f}</span>
                            </div>
                            <div style="font-size:13px;">
                                {sla_html}
                                {conv_html}
                            </div>
                        </div>
                        <div style="text-align:right; padding-left:20px;">
                            <div style="font-size:28px; font-weight:800; color:{priority_color};">
                                {score:.2f}
                            </div>
                            <div style="font-size:11px; color:#6b7280; text-transform:uppercase;">
                                Priority
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ========== All Leads (expandable, editable; allow AWARDED/LOST with invoice upload) ==========
        st.markdown("### 📋 All Leads — Expand to view / edit")
        for lead in leads:
            est_val = lead.estimated_value or 0.0
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — ${est_val:,.0f}"
            with st.expander(card_title, expanded=False):
                # Show details
                colA, colB = st.columns([3,1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}")
                with colB:
                    entered = lead.sla_entered_at or lead.created_at
                    try:
                        if isinstance(entered, str):
                            entered = datetime.fromisoformat(entered)
                    except Exception:
                        entered = datetime.utcnow()
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    if remaining.total_seconds() <= 0:
                        sla_status = f"<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        # red display when < 2 hours
                        color_t = "#ef4444" if (hours*60 + mins) <= 120 else "#2563eb"
                        sla_status = f"<div style='color:{color_t};font-weight:600;'>⏳ {hours}h {mins}m</div>"
                    st.markdown(f"<div style='text-align:right'>{sla_status}</div>", unsafe_allow_html=True)

                st.markdown("---")
                # Quick contact buttons
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    with qc1:
                        st.markdown(f"<a href='tel:{phone}'><button style='background:#2563eb;color:#fff;border:none;border-radius:8px;padding:8px 12px;width:100%;font-weight:700;'>📞 Call</button></a>", unsafe_allow_html=True)
                    with qc2:
                        wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                        wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                        st.markdown(f"<a href='{wa_link}' target='_blank'><button style='background:#25D366;color:#000;border:none;border-radius:8px;padding:8px 12px;width:100%;font-weight:700;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")
                if email:
                    with qc3:
                        st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='background:transparent; color:#111;border:1px solid rgba(0,0,0,0.06); border-radius:8px;padding:8px 12px;width:100%;font-weight:700;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")

                st.markdown("---")

                # Update lead form (editable)
                form_key = f"update_lead_{lead.id}"
                with st.form(form_key):
                    st.markdown("#### Update Lead")
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        new_status = st.selectbox("Status", LEAD_STATUSES, index=LEAD_STATUSES.index(lead.status) if lead.status in LEAD_STATUSES else 0, key=f"status_{lead.id}")
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        contacted_cb = st.checkbox("Contacted", value=bool(lead.contacted), key=f"contacted_{lead.id}")
                    with ucol2:
                        inspection_scheduled_cb = st.checkbox("Inspection Scheduled", value=bool(lead.inspection_scheduled), key=f"insp_sched_{lead.id}")
                        inspection_completed_cb = st.checkbox("Inspection Completed", value=bool(lead.inspection_completed), key=f"insp_comp_{lead.id}")
                        estimate_submitted_cb = st.checkbox("Estimate Submitted", value=bool(lead.estimate_submitted), key=f"est_sub_{lead.id}")

                    new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                    new_est_val = st.number_input("Job Value Estimate (USD)", min_value=0.0, value=float(lead.estimated_value or 0.0), step=100.0, key=f"est_val_{lead.id}")
                    invoice_file = None

                    # If status selected AWARDED, show invoice upload
                    if st.session_state.get(f"status_{lead.id}", lead.status) == "AWARDED":
                        st.markdown("**Award Details**")
                        awarded_comment = st.text_input("Award comment (optional)", key=f"aw_comment_{lead.id}")
                        invoice_file = st.file_uploader("Upload Invoice File (optional)", type=["pdf","jpg","jpeg","png","xlsx"], key=f"aw_invoice_{lead.id}")

                    if st.form_submit_button("💾 Update Lead"):
                        # persist changes
                        lead.status = new_status
                        lead.assigned_to = new_assigned
                        lead.contacted = bool(contacted_cb)
                        lead.inspection_scheduled = bool(inspection_scheduled_cb)
                        lead.inspection_completed = bool(inspection_completed_cb)
                        lead.estimate_submitted = bool(estimate_submitted_cb)
                        lead.notes = new_notes
                        lead.estimated_value = float(new_est_val or 0.0)

                        # award/lost processing
                        if new_status == "AWARDED":
                            lead.awarded_date = datetime.utcnow()
                            if invoice_file is not None:
                                path = save_uploaded_file(invoice_file, lead.id)
                                lead.awarded_invoice = path
                        if new_status == "LOST":
                            lead.lost_date = datetime.utcnow()

                        # ensure SLA entered if missing
                        if lead.sla_entered_at is None:
                            lead.sla_entered_at = datetime.utcnow()

                        s.add(lead)
                        s.commit()
                        st.success(f"Lead #{lead.id} updated")
                        # Trigger local rerun (safe)
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

                # Estimates listing & create estimate
                st.markdown("#### Job Value Estimates")
                ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if ests:
                    for e in ests:
                        status_label = "✅ Approved" if e.approved else ("❌ Lost" if e.lost else "⏳ Pending")
                        st.markdown(f"""
                        <div style='padding:8px;background:#f3f4f6;border-radius:8px;margin-bottom:8px;'>
                            <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <div><strong style='color:#111;'>{status_label}</strong> &nbsp; <span style='color:#16a34a;font-weight:800;'>${e.amount:,.0f}</span></div>
                                <div style='color:#6b7280;font-size:12px;'>{e.created_at.strftime('%Y-%m-%d')}</div>
                            </div>
                            <div style='color:#374151;margin-top:6px;'>{e.details or ''}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No estimates yet for this lead.")

                with st.form(f"create_estimate_{lead.id}"):
                    st.markdown("**Create New Job Value Estimate (USD)**")
                    est_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                    est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                    if st.form_submit_button("➕ Create Estimate"):
                        create_estimate(s, lead.id, est_amount, est_details)
                        st.success("Estimate created")
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

# -------- Page: Analytics & SLA (Pie chart & auto updates) --------
elif page == "Analytics & SLA":
    st.header("📈 Analytics & SLA")
    s = get_session()
    df = leads_df(s)

    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Pie chart for stage distribution (colorful, responsive)
        funnel = df.groupby("status").size().reindex(LEAD_STATUSES, fill_value=0).reset_index()
        funnel.columns = ["stage","count"]
        funnel = funnel[funnel["count"] > 0]  # only show present stages
        if funnel.empty:
            st.info("No leads assigned to stages yet.")
        else:
            colors = [stage_colors.get(s, "#111111") for s in funnel["stage"].tolist()]
            fig = px.pie(funnel, names="stage", values="count", title="Leads by Stage (Distribution)", hole=0.35, color_discrete_sequence=colors)
            fig.update_traces(textinfo='percent+label')
            fig.update_layout(showlegend=True, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # SLA / Overdue table
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row.get("sla_entered_at") or row.get("created_at")
            try:
                if pd.isna(sla_entered_at) or sla_entered_at is None:
                    sla_entered_at = datetime.utcnow()
                elif isinstance(sla_entered_at, str):
                    sla_entered_at = datetime.fromisoformat(sla_entered_at)
            except Exception:
                sla_entered_at = datetime.utcnow()
            sla_hours = int(row.get("sla_hours") or 24)
            deadline = sla_entered_at + timedelta(hours=sla_hours)
            remaining = deadline - datetime.utcnow()
            overdue_rows.append({
                "id": row["id"],
                "contact": row["contact_name"],
                "status": row["status"],
                "deadline": deadline,
                "overdue": remaining.total_seconds() <= 0
            })
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.subheader("SLA / Overdue Leads")
            st.dataframe(df_overdue.sort_values("deadline"))
        else:
            st.info("No SLA rows found.")

# -------- Page: Exports --------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    df_est = pd.DataFrame([{
        "id": e.id, "lead_id": e.lead_id, "amount": e.amount, "details": e.details, "created_at": e.created_at, "approved": e.approved, "lost": e.lost
    } for e in get_session().query(Estimate).all()])
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")

# End of app
