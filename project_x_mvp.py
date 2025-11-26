# project_x_mvp_fixed.py
"""
Project X — Single-file Streamlit app (fixed & enhanced)
- Single-file app with SQLite + SQLAlchemy
- Google Ads style Pipeline Dashboard (2 rows x 4 cols KPI)
- Editable leads, Awarded -> allow invoice upload
- Job Value Estimate naming
- Priority Top 8 fixed display
- Analytics: responsive pie chart auto-updating
- White background UI
- No st.experimental_rerun() calls
"""

import os
import io
from datetime import datetime, timedelta
import sqlite3
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Try optional joblib (safe)
try:
    import joblib
except Exception:
    joblib = None

# -----------------------
# Config
# -----------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_fixed.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_invoices")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


# -----------------------
# Models
# -----------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="")
    source_details = Column(String, default="")
    contact_name = Column(String, default="")
    contact_phone = Column(String, default="")
    contact_email = Column(String, default="")
    property_address = Column(String, default="")
    damage_type = Column(String, default="other")
    assigned_to = Column(String, default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    status = Column(String, default="NEW")  # statuses in LEAD_STATUSES
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimated_value = Column(Float, default=0.0)

    # Awarded / Lost extras
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    lost_comment = Column(Text, nullable=True)

    estimates = relationship("Estimate", back_populates="lead")


class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    amount = Column(Float, default=0.0)
    details = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)

    lead = relationship("Lead", back_populates="estimates")


def init_db():
    Base.metadata.create_all(bind=engine)


# -----------------------
# Lead Statuses
# -----------------------
LEAD_STATUSES = [
    "NEW",
    "CONTACTED",
    "INSPECTION_SCHEDULED",
    "INSPECTION_COMPLETED",
    "ESTIMATE_SUBMITTED",
    "AWARDED",
    "LOST"
]

# map colors
STAGE_COLORS = {
    "NEW": "#2563eb",
    "CONTACTED": "#eab308",
    "INSPECTION_SCHEDULED": "#f97316",
    "INSPECTION_COMPLETED": "#14b8a6",
    "ESTIMATE_SUBMITTED": "#a855f7",
    "AWARDED": "#22c55e",
    "LOST": "#ef4444",
}


# -----------------------
# Helpers
# -----------------------
def get_session():
    return SessionLocal()


def leads_df(session):
    leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
    rows = []
    for l in leads:
        rows.append({
            "id": l.id,
            "source": l.source,
            "source_details": l.source_details,
            "contact_name": l.contact_name,
            "contact_phone": l.contact_phone,
            "contact_email": l.contact_email,
            "property_address": l.property_address,
            "damage_type": l.damage_type,
            "assigned_to": l.assigned_to,
            "notes": l.notes,
            "created_at": l.created_at,
            "sla_hours": l.sla_hours,
            "sla_entered_at": l.sla_entered_at,
            "status": l.status,
            "contacted": l.contacted,
            "inspection_scheduled": l.inspection_scheduled,
            "inspection_completed": l.inspection_completed,
            "estimate_submitted": l.estimate_submitted,
            "estimated_value": l.estimated_value,
            "awarded_date": l.awarded_date,
            "awarded_invoice": l.awarded_invoice,
            "awarded_comment": l.awarded_comment,
            "lost_date": l.lost_date,
            "lost_comment": l.lost_comment,
        })
    return pd.DataFrame(rows)


def save_uploaded_file(uploaded_file, lead_id):
    if uploaded_file is None:
        return None
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_FOLDER, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def format_currency(val):
    try:
        return "${:,.0f}".format(float(val))
    except Exception:
        return "$0"


def compute_priority_for_lead_row(row, weights):
    # Simple heuristic: normalize estimated value, SLA urgency, and flags
    try:
        value = float(row.get("estimated_value") or 0.0)
    except:
        value = 0.0
    value_baseline = float(weights.get("value_baseline", 5000.0))
    value_score = min(1.0, value / max(1.0, value_baseline))

    # SLA: how close to deadline
    sla_entered = row.get("sla_entered_at") or row.get("created_at")
    if sla_entered is None:
        time_left_h = 9999.0
    else:
        if isinstance(sla_entered, str):
            try:
                sla_entered = datetime.fromisoformat(sla_entered)
            except:
                sla_entered = datetime.utcnow()
        deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
        time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)

    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)  # closer deadlines -> higher score

    contacted_flag = 0.0 if bool(row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(row.get("estimate_submitted")) else 1.0

    urgency_component = (contacted_flag * weights.get("contacted_w", 0.6)
                        + inspection_flag * weights.get("inspection_w", 0.5)
                        + estimate_flag * weights.get("estimate_w", 0.5))

    total_weight = (weights.get("value_weight", 0.5)
                   + weights.get("sla_weight", 0.35)
                   + weights.get("urgency_weight", 0.15))
    if total_weight <= 0:
        total_weight = 1.0
    score = (value_score * weights.get("value_weight", 0.5)
            + sla_score * weights.get("sla_weight", 0.35)
            + urgency_component * weights.get("urgency_weight", 0.15)) / total_weight
    score = max(0.0, min(score, 1.0))
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_h


# -----------------------
# App CSS (white background)
# -----------------------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg: #ffffff;
  --muted: #6b7280;
  --white: #ffffff;
  --black-card: #0b0b0b;
  --radius: 10px;
  --money-green: #16a34a;
}
body, .stApp {
  background: var(--bg);
  color: #111827;
  font-family: 'Roboto', sans-serif;
}
.header { padding: 12px; color: #111827; font-weight:700; font-size:20px; }
.card {
  background: var(--black-card);
  color: var(--white);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 12px;
  border: 1px solid rgba(0,0,0,0.12);
}
.kpi-card {
  background: linear-gradient(180deg, rgba(0,0,0,0.9), rgba(0,0,0,0.75));
  color: white;
  border-radius: 12px;
  padding: 18px;
  margin: 8px;
}
.small-muted { color: var(--muted); font-size:12px; }
.progress-bar { width:100%; height:10px; background: #e6e6e6; border-radius:6px; overflow:hidden; }
.progress-fill { height:100%; border-radius:6px; transition: width 0.3s; }
.quick-btn { padding:8px 12px; border-radius:8px; font-weight:700; cursor:pointer;}
"""

# -----------------------
# Initialize DB
# -----------------------
init_db()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Project X — Pipeline", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar controls
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
    w = st.session_state.weights
    w["value_weight"] = st.slider("Estimate value weight", 0.0, 1.0, float(w["value_weight"]), step=0.05)
    w["sla_weight"] = st.slider("SLA urgency weight", 0.0, 1.0, float(w["sla_weight"]), step=0.05)
    w["urgency_weight"] = st.slider("Flags urgency weight", 0.0, 1.0, float(w["urgency_weight"]), step=0.05)
    st.markdown("Within urgency flags:")
    w["contacted_w"] = st.slider("Not-contacted weight", 0.0, 1.0, float(w["contacted_w"]), step=0.05)
    w["inspection_w"] = st.slider("Not-scheduled weight", 0.0, 1.0, float(w["inspection_w"]), step=0.05)
    w["estimate_w"] = st.slider("No-estimate weight", 0.0, 1.0, float(w["estimate_w"]), step=0.05)
    w["value_baseline"] = st.number_input("Value baseline", min_value=100.0, value=float(w["value_baseline"]), step=100.0)
    st.markdown("---")
    st.checkbox("Auto-refresh pipeline every 30s", key="autorefresh_toggle")
    st.markdown(f"<small class='small-muted'>DB file: {DB_FILE}</small>", unsafe_allow_html=True)

# Optionally inject JS auto-refresh if user enabled it
if st.session_state.get("autorefresh_toggle", False):
    # small invisible HTML that reloads page every 30s
    st.components.v1.html("<script>setInterval(()=>location.reload(),30000);</script>", height=0)


# -----------------------
# Page: Lead Capture
# -----------------------
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
        submitted = st.form_submit_button("Create Lead")
        if submitted:
            s = get_session()
            new = Lead(
                source=source, source_details=source_details, contact_name=contact_name,
                contact_phone=contact_phone, contact_email=contact_email,
                property_address=property_address, damage_type=damage_type,
                assigned_to=assigned_to, notes=notes,
                sla_hours=int(sla_hours), sla_entered_at=datetime.utcnow(),
                status="NEW", estimated_value=0.0,
                contacted=(qualified_choice == "Yes")
            )
            s.add(new)
            s.commit()
            st.success(f"Lead created (ID: {new.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))


# -----------------------
# Page: Pipeline Board
# -----------------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Google Ads style")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # Attempt to load optional model (safe)
        lead_model = None
        try:
            if joblib is not None and os.path.exists("lead_conversion_model.pkl"):
                lead_model = joblib.load("lead_conversion_model.pkl")
        except Exception:
            lead_model = None

        # Calculate KPIs
        total_leads = len(df)
        qualified_leads = len(df[df.get("contacted", False) == True])
        total_value = df["estimated_value"].sum() if "estimated_value" in df and not df.empty else 0.0
        awarded_leads = len(df[df["status"] == "AWARDED"]) if "status" in df else 0
        lost_leads = len(df[df["status"] == "LOST"]) if "status" in df else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads

        # KPI cards: 2 rows x 4 columns (unique colors)
        st.markdown("### 📊 Key Performance Indicators")
        # Build 8 KPI items (you can customize)
        kpis = [
            {"label": "Total Leads", "value": total_leads, "sub": f"{qualified_leads} Qualified", "color": "#2563eb"},
            {"label": "Pipeline Value", "value": format_currency(total_value), "sub": "Estimated", "color": "#16a34a"},
            {"label": "Conversion Rate", "value": f"{conversion_rate:.1f}%", "sub": f"{awarded_leads}/{closed_leads} Won", "color": "#7c3aed"},
            {"label": "Active Leads", "value": active_leads, "sub": "In Progress", "color": "#f97316"},
            {"label": "Awarded Jobs", "value": awarded_leads, "sub": "Jobs Awarded", "color": "#10b981"},
            {"label": "Lost Jobs", "value": lost_leads, "sub": "Jobs Lost", "color": "#ef4444"},
            {"label": "Avg Estimate", "value": format_currency(df['estimated_value'].mean() if not df.empty else 0), "sub": "Average", "color": "#0ea5e9"},
            {"label": "Open Estimates", "value": len(df[df['estimate_submitted'] == False]) if 'estimate_submitted' in df else 0, "sub": "Need Estimates", "color": "#f43f5e"},
        ]

        # First row (4 cols)
        cols1 = st.columns(4)
        for i in range(4):
            k = kpis[i]
            with cols1[i]:
                st.markdown(f"""
                <div class="kpi-card" style="border-left:6px solid {k['color']};">
                    <div style="font-size:12px; font-weight:700; color:#e5e7eb;">{k['label']}</div>
                    <div style="font-size:28px; font-weight:800; color:white; margin-top:6px;">{k['value']}</div>
                    <div style="margin-top:6px; color:#d1d5db; font-size:12px;">{k['sub']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Second row (4 cols)
        cols2 = st.columns(4)
        for i in range(4, 8):
            k = kpis[i]
            with cols2[i - 4]:
                st.markdown(f"""
                <div class="kpi-card" style="border-left:6px solid {k['color']};">
                    <div style="font-size:12px; font-weight:700; color:#e5e7eb;">{k['label']}</div>
                    <div style="font-size:28px; font-weight:800; color:white; margin-top:6px;">{k['value']}</div>
                    <div style="margin-top:6px; color:#d1d5db; font-size:12px;">{k['sub']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Pipeline Stage Cards (arranged horizontally via columns)
        st.markdown("### 📈 Pipeline Stages")
        stage_counts = df["status"].value_counts().to_dict() if not df.empty else {}
        # show stages in one row of columns
        cols = st.columns(len(LEAD_STATUSES))
        for i, stg in enumerate(LEAD_STATUSES):
            count = stage_counts.get(stg, 0)
            pct = (count / total_leads * 100) if total_leads > 0 else 0.0
            color = STAGE_COLORS.get(stg, "#000000")
            with cols[i]:
                st.markdown(f"""
                <div class="card" style="background: {STAGE_COLORS.get(stg,'#111')}; color: white;">
                    <div style="font-weight:700; font-size:12px;">{stg.replace('_', ' ').title()}</div>
                    <div style="font-size:22px; font-weight:800; margin-top:6px; color: #ffffff;">{count}</div>
                    <div class="progress-bar" style="margin-top:8px;">
                        <div class="progress-fill" style="background:{color}; width:{pct}%;"></div>
                    </div>
                    <div style="text-align:center; margin-top:8px; font-size:12px; color: #e5e7eb;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Priority Leads (Top 8)
        st.markdown("### 🎯 Priority Leads (Top 8)")
        priority_list = []
        for _, row in df.iterrows():
            score, *_ = compute_priority_for_lead_row(row, weights)
            # SLA calculation robust
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if sla_entered is None:
                sla_entered = datetime.utcnow()
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0
            hours_left = int(max(0, remaining.total_seconds() // 3600))
            mins_left = int(max(0, (remaining.total_seconds() % 3600) // 60))
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "hours_left": hours_left,
                "mins_left": mins_left,
                "damage_type": row.get("damage_type", "Unknown")
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        if pr_df.empty:
            st.info("No priority leads to display.")
        else:
            # display top 8
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"
                    priority_label = "🟢 NORMAL"

                status_color = STAGE_COLORS.get(r["status"], "#111111")
                sla_html = ""
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
                else:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {r['hours_left']}h {r['mins_left']}m left</span>"

                est_str = format_currency(r["estimated_value"])
                # Render
                st.markdown(f"""
                <div class="card" style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                        <div style="margin-bottom:6px;">
                            <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                            <span style="background:{status_color}22; color:{status_color}; padding:6px 10px; border-radius:20px; margin-left:8px; font-weight:700;">
                                {r['status']}
                            </span>
                        </div>
                        <div style="font-size:16px; font-weight:800; color:#ffffff;">#{int(r['id'])} — {r['contact_name']}</div>
                        <div style="font-size:13px; color:#d1d5db; margin-top:6px;">
                            {r['damage_type'].title()} | Est: <span style="color:#22c55e; font-weight:800;">{est_str}</span>
                        </div>
                        <div style="margin-top:8px; font-size:13px; color:#d1d5db;">
                            {sla_html}
                        </div>
                    </div>
                    <div style="text-align:right; padding-left:20px;">
                        <div style="font-size:28px; font-weight:800; color:{priority_color};">{score:.2f}</div>
                        <div style="font-size:11px; color:var(--muted); text-transform:uppercase;">Priority</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed Lead Cards (Expandable, editable)
        st.markdown("### 📋 All Leads (expand a card to edit / change status)")
        for lead in leads:
            est_val = lead.estimated_value or 0
            est_val_display = format_currency(est_val)
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}")
                with colB:
                    entered = lead.sla_entered_at or lead.created_at or datetime.utcnow()
                    if isinstance(entered, str):
                        try:
                            entered = datetime.fromisoformat(entered)
                        except:
                            entered = datetime.utcnow()
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    if remaining.total_seconds() <= 0:
                        sla_status_html = f"<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        # show red if less than 1 hour
                        time_color = "#ef4444" if hours == 0 else "#2563eb"
                        sla_status_html = f"<div style='color:{time_color};font-weight:700;'>⏳ {hours}h {mins}m</div>"
                    st.markdown(f"""
                    <div style="text-align:right;">
                        <div style="margin-bottom:8px;">
                            <span style="padding:6px 12px; border-radius:18px; background:{STAGE_COLORS.get(lead.status,'#111')}33; color:{STAGE_COLORS.get(lead.status,'#111')}; font-weight:700;">
                                {lead.status}
                            </span>
                        </div>
                        {sla_status_html}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact buttons
                qc1, qc2, qc3, qc4 = st.columns([1, 1, 1, 4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    qc1.markdown(f"<a href='tel:{phone}'><button style='background:#2563eb;color:#fff;border:none;border-radius:8px;padding:8px 10px;width:100%;font-weight:700;'>📞 Call</button></a>", unsafe_allow_html=True)
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20request."
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button style='background:#25D366;color:#000;border:none;border-radius:8px;padding:8px 10px;width:100%;font-weight:700;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(""); qc2.write("")
                if email:
                    qc3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='background:transparent; color:#111;border:1px solid rgba(0,0,0,0.08);border-radius:8px;padding:8px 10px;width:100%;font-weight:700;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write("")
                qc4.write("")

                st.markdown("---")

                # Update Lead form
                with st.form(f"update_lead_{lead.id}"):
                    st.markdown("#### Update Lead")
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        new_status = st.selectbox("Status", LEAD_STATUSES, index=LEAD_STATUSES.index(lead.status), key=f"status_{lead.id}")
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        contacted = st.checkbox("Contacted", value=lead.contacted, key=f"contacted_{lead.id}")
                    with ucol2:
                        inspection_scheduled = st.checkbox("Inspection Scheduled", value=lead.inspection_scheduled, key=f"insp_sched_{lead.id}")
                        inspection_completed = st.checkbox("Inspection Completed", value=lead.inspection_completed, key=f"insp_comp_{lead.id}")
                        estimate_submitted = st.checkbox("Estimate Submitted", value=lead.estimate_submitted, key=f"est_sub_{lead.id}")
                    new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")

                    # Awarded-specific fields (file uploader + comment)
                    awarded_invoice_uploader = None
                    awarded_comment_text = None
                    lost_comment_text = None
                    if new_status == "AWARDED":
                        st.markdown("**Job Awarded — Upload Invoice (optional)**")
                        awarded_invoice_uploader = st.file_uploader(f"Upload invoice for lead {lead.id}", type=["pdf", "jpg", "png", "xlsx"], key=f"award_file_{lead.id}")
                        awarded_comment_text = st.text_area("Awarded comment (optional)", key=f"award_comment_{lead.id}")
                    if new_status == "LOST":
                        st.markdown("**Job Lost — Add comment**")
                        lost_comment_text = st.text_area("Lost comment", key=f"lost_comment_{lead.id}")

                    if st.form_submit_button("💾 Update Lead"):
                        # persist updates
                        lead.status = new_status
                        lead.assigned_to = new_assigned
                        lead.contacted = contacted
                        lead.inspection_scheduled = inspection_scheduled
                        lead.inspection_completed = inspection_completed
                        lead.estimate_submitted = estimate_submitted
                        lead.notes = new_notes

                        if new_status == "AWARDED":
                            lead.awarded_date = datetime.utcnow()
                            if awarded_comment_text:
                                lead.awarded_comment = awarded_comment_text
                            if awarded_invoice_uploader is not None:
                                saved = save_uploaded_file(awarded_invoice_uploader, lead.id)
                                if saved:
                                    lead.awarded_invoice = saved
                        if new_status == "LOST":
                            lead.lost_date = datetime.utcnow()
                            if lost_comment_text:
                                lead.lost_comment = lost_comment_text

                        # If estimate_submitted changed and estimated_value input not set, keep same
                        s.add(lead)
                        s.commit()
                        st.success(f"Lead #{lead.id} updated!")

                st.markdown("---")

                # Estimates section
                st.markdown("#### 💰 Job Value Estimate (USD)")
                lead_estimates = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if lead_estimates:
                    for est in lead_estimates:
                        est_status = "✅ Approved" if est.approved else ("❌ Lost" if est.lost else "⏳ Pending")
                        est_color = "#22c55e" if est.approved else ("#ef4444" if est.lost else "#f97316")
                        st.markdown(f"""
                        <div style="padding:10px;background:rgba(255,255,255,0.03);border-radius:8px;margin:8px 0;">
                          <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                              <span style="color:{est_color};font-weight:700;">{est_status}</span>
                              <span style="margin-left:12px;color:#22c55e;font-weight:700;font-size:18px;">{format_currency(est.amount)}</span>
                            </div>
                            <div style="color:#6b7280;font-size:12px;">{est.created_at.strftime('%Y-%m-%d')}</div>
                          </div>
                          <div style="margin-top:6px;color:#374151;">{est.details}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No estimates yet.")

                # Create estimate form
                with st.form(f"create_estimate_{lead.id}"):
                    st.markdown("**Create New Job Value Estimate (USD)**")
                    est_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                    est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                    if st.form_submit_button("➕ Create Estimate"):
                        new_est = Estimate(lead_id=lead.id, amount=float(est_amount or 0.0), details=est_details)
                        lead.estimate_submitted = True
                        # Optionally set lead.estimated_value to this amount (update)
                        if est_amount and est_amount > 0:
                            lead.estimated_value = float(est_amount)
                        s.add(new_est)
                        s.add(lead)
                        s.commit()
                        st.success("Estimate created!")

# -----------------------
# Page: Analytics & SLA (pie chart)
# -----------------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Pie chart for stages (colorful & responsive)
        funnel = df.groupby("status").size().reindex(LEAD_STATUSES, fill_value=0).reset_index()
        funnel.columns = ["stage", "count"]
        funnel = funnel[funnel["count"] > 0]  # hide zero counts for pie

        if funnel.empty:
            st.info("No leads to show on pie chart.")
        else:
            colors = [STAGE_COLORS.get(s, "#111111") for s in funnel["stage"].tolist()]
            fig = px.pie(funnel, names="stage", values="count", title="Leads by Stage", color="stage", 
                         color_discrete_map={r["stage"]: STAGE_COLORS.get(r["stage"], "#111") for _, r in funnel.iterrows()})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=450)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row.get("sla_entered_at") or row.get("created_at") or datetime.utcnow()
            if isinstance(sla_entered_at, str):
                try:
                    sla_entered_at = datetime.fromisoformat(sla_entered_at)
                except:
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
            st.dataframe(df_overdue.sort_values("deadline"))
        else:
            st.info("No SLA overdue leads.")

# -----------------------
# Page: Exports
# -----------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    # estimates
    ests = session = get_session().query(Estimate).all()
    if ests:
        df_est = pd.DataFrame([{
            "id": e.id, "lead_id": e.lead_id, "amount": e.amount, "details": e.details, "created_at": e.created_at, "approved": e.approved
        } for e in ests])
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")
    else:
        st.info("No estimates to export.")
