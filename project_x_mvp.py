# project_x_singlefile.py
# Single-file Streamlit app (Option A)
# Assan / Project X — Pipeline + Analytics (combined, self-contained)

import os
from datetime import datetime, timedelta, time as dtime
import sqlite3
import tempfile
import traceback

import pandas as pd
import streamlit as st
import plotly.express as px

# SQLAlchemy is used for a nicer model layer
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Try import joblib (optional model). If missing, continue without model.
try:
    import joblib
except Exception:
    joblib = None

# ---------------------------
# CONFIG
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_singlefile.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_PATH = "lead_conversion_model.pkl"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------- Lead statuses ----------
class LeadStatus:
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    INSPECTION_SCHEDULED = "INSPECTION_SCHEDULED"
    INSPECTION_COMPLETED = "INSPECTION_COMPLETED"
    ESTIMATE_SUBMITTED = "ESTIMATE_SUBMITTED"
    AWARDED = "AWARDED"
    LOST = "LOST"

    ALL = [
        NEW,
        CONTACTED,
        INSPECTION_SCHEDULED,
        INSPECTION_COMPLETED,
        ESTIMATE_SUBMITTED,
        AWARDED,
        LOST
    ]

# ---------------------------
# MODELS
# ---------------------------
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="Unknown")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, nullable=True)
    status = Column(String, default=LeadStatus.NEW)
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)

    # Flags
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)

    # Awarded/Lost fields
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)

    lost_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)

    qualified = Column(Boolean, default=False)


class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, nullable=False)
    amount = Column(Float, default=0.0)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)


# ---------------------------
# DATABASE UTILITIES
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()


def add_lead(session,
             source="Unknown", source_details=None, contact_name=None, contact_phone=None, contact_email=None,
             property_address=None, damage_type=None, assigned_to=None, notes=None, sla_hours=24, qualified=False,
             estimated_value=None):
    lead = Lead(
        source=source, source_details=source_details, contact_name=contact_name,
        contact_phone=contact_phone, contact_email=contact_email,
        property_address=property_address, damage_type=damage_type, assigned_to=assigned_to,
        notes=notes, sla_hours=int(sla_hours), qualified=bool(qualified),
        estimated_value=float(estimated_value) if estimated_value else None,
        sla_entered_at=datetime.utcnow()
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead


def create_estimate(session, lead_id, amount, details):
    est = Estimate(lead_id=lead_id, amount=float(amount), details=details)
    session.add(est)
    session.commit()
    session.refresh(est)
    # mark lead as estimate_submitted
    lead = session.query(Lead).filter(Lead.id == lead_id).first()
    if lead:
        lead.estimate_submitted = True
        lead.estimate_submitted_at = datetime.utcnow()
        lead.status = LeadStatus.ESTIMATE_SUBMITTED
        session.add(lead)
        session.commit()
    return est


def leads_df(session):
    rows = session.query(Lead).all()
    data = []
    for r in rows:
        data.append({
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
            "estimated_value": r.estimated_value or 0.0,
            "status": r.status,
            "created_at": r.created_at,
            "sla_hours": r.sla_hours or 24,
            "sla_entered_at": r.sla_entered_at,
            "contacted": bool(r.contacted),
            "inspection_scheduled": bool(r.inspection_scheduled),
            "inspection_scheduled_at": r.inspection_scheduled_at,
            "inspection_completed": bool(r.inspection_completed),
            "estimate_submitted": bool(r.estimate_submitted),
            "awarded_date": r.awarded_date,
            "awarded_invoice": r.awarded_invoice,
            "lost_date": r.lost_date,
            "qualified": bool(r.qualified),
        })
    df = pd.DataFrame(data)
    if df.empty:
        # Ensure columns exist
        df = pd.DataFrame(columns=[
            "id", "source", "source_details", "contact_name", "contact_phone", "contact_email",
            "property_address", "damage_type", "assigned_to", "notes", "estimated_value",
            "status", "created_at", "sla_hours", "sla_entered_at", "contacted",
            "inspection_scheduled", "inspection_scheduled_at", "inspection_completed",
            "estimate_submitted", "awarded_date", "awarded_invoice", "lost_date", "qualified"
        ])
    return df


def estimates_df(session):
    rows = session.query(Estimate).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "lead_id": r.lead_id,
            "amount": r.amount,
            "details": r.details,
            "created_at": r.created_at,
            "approved": bool(r.approved),
            "lost": bool(r.lost)
        })
    return pd.DataFrame(data)


# ---------------------------
# UTILITIES
# ---------------------------
def save_uploaded_file(uploaded_file, prefix="file"):
    if uploaded_file is None:
        return None
    folder = os.path.join(os.getcwd(), "uploaded_files")
    os.makedirs(folder, exist_ok=True)
    fname = f"{prefix}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def compute_priority_for_lead_row(lead_row, weights):
    """
    Returns: score (0..1), value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_hours
    """
    try:
        value = float(lead_row.get("estimated_value") or 0.0)
        baseline = float(weights.get("value_baseline", 5000.0))
        value_score = min(1.0, value / max(1.0, baseline))

        sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
        if sla_entered is None:
            time_left_h = 9999.0
        else:
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            try:
                deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
                time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
            except Exception:
                time_left_h = 9999.0
    except Exception:
        value_score = 0.0
        time_left_h = 9999.0

    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)
    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0

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


def predict_lead_probability(lead_model, lead_row):
    """ If a model exists, make a safe prediction. This is a stub — adapt to your model features. """
    if lead_model is None:
        return None
    try:
        # Example: expect model to accept DataFrame row -> prob
        X = pd.DataFrame([{
            "estimated_value": lead_row.get("estimated_value") or 0.0,
            "qualified": 1 if lead_row.get("qualified") else 0,
            "sla_hours": lead_row.get("sla_hours") or 24
        }])
        if hasattr(lead_model, "predict_proba"):
            p = lead_model.predict_proba(X)[:, 1][0]
            return float(p)
        # fallback
        return float(lead_model.predict(X)[0])
    except Exception:
        return None


# ---------------------------
# UI / CSS
# ---------------------------
WHITE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#ffffff;
  --muted:#6b7280;
  --white:#ffffff;
  --text:#0b0f13;
  --placeholder:#9ca3af;
  --radius:10px;
  --primary-red:#ef4444;
  --money-green:#22c55e;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}

/* Base */
body, .stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Roboto', sans-serif;
}

/* Header */
.header { padding: 12px; color: var(--text); font-weight:600; font-size:18px; }

/* Sidebar */
section[data-testid="stSidebar"] {
  padding: 18px;
  background: #f8fafc !important;
}

/* Metric card */
.metric-card {
    border-radius: 10px;
    padding: 16px;
    margin: 6px;
    color: var(--white);
}
.stage-card {
    background: #000000;
    color: #ffffff;
    padding: 12px;
    border-radius: 8px;
    margin: 6px;
}
.small-muted { color: var(--muted); font-size:12px; }
.progress-bar {
    width:100%; height:8px; background: #e6e6e6; border-radius:6px; overflow:hidden; margin-top:8px;
}
.progress-fill { height:100%; border-radius:6px; transition:width 0.3s ease; }

/* Quickly style buttons inside HTML */
.btn-inline { padding:6px 10px; border-radius:8px; font-weight:600; border:none; cursor:pointer; }
"""

# ---------------------------
# APP START
# ---------------------------
st.set_page_config(page_title="Assan — CRM (Single File)", layout="wide", initial_sidebar_state="expanded")
init_db()
st.markdown(f"<style>{WHITE_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div class='header'>Assan — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar controls + weight tuning
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

    if "pipeline_autorefresh" not in st.session_state:
        st.session_state.pipeline_autorefresh = True
    st.session_state.pipeline_autorefresh = st.checkbox("Auto-refresh pipeline (30s)", value=st.session_state.pipeline_autorefresh)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", source_details="gclid=demo",
                 contact_name="Demo Customer", contact_phone="+15550000", contact_email="demo@example.com",
                 property_address="100 Demo Ave", damage_type="water",
                 assigned_to="Alex", estimated_value=4500, notes="Demo lead", sla_hours=24, qualified=True)
        st.success("Demo lead added")

# Inject JS auto-refresh (safe) when enabled
if st.session_state.get("pipeline_autorefresh", False):
    # Only reload every 30s — this triggers a full page reload
    st.components.v1.html(
        """
        <script>
        // reload page every 30s
        setTimeout(()=>{ window.location.reload(); }, 30000);
        </script>
        """,
        height=0
    )

# Load model safely
lead_model = None
if joblib is not None and os.path.exists(MODEL_PATH):
    try:
        lead_model = joblib.load(MODEL_PATH)
    except Exception:
        lead_model = None

# ---------------------------
# Page: Leads / Capture
# ---------------------------
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
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
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
                qualified=True if qualified_choice == "Yes" else False,
                estimated_value=float(estimated_value or 0.0)
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


# ---------------------------
# Page: Pipeline Board (Google Ads style)
# ---------------------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Google Ads style")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # KPI cards (2 rows x 4 columns) — unique colors
        total_leads = len(df)
        qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
        total_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
        awarded_leads = int(df[df["status"] == LeadStatus.AWARDED].shape[0]) if not df.empty else 0
        lost_leads = int(df[df["status"] == LeadStatus.LOST].shape[0]) if not df.empty else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads
        avg_sla_hours = 0
        try:
            avg_sla_hours = float(df["sla_hours"].mean()) if not df.empty else 0.0
        except Exception:
            avg_sla_hours = 0.0
        avg_value = (total_value / total_leads) if total_leads else 0.0

        # Colors for KPI cards
        KPI_COLORS = [
            ("#2563eb", "Total Leads", total_leads, f"↑ {qualified_leads} qualified"),
            ("#a855f7", "Pipeline Value", f"${total_value:,.0f}", "Estimated value"),
            ("#22c55e", "Conversion Rate", f"{conversion_rate:.1f}%", f"{awarded_leads}/{closed_leads} won"),
            ("#f97316", "Active Leads", active_leads, "In progress"),
            ("#ef4444", "Awarded", awarded_leads, "Won jobs"),
            ("#6d28d9", "Lost", lost_leads, "Lost jobs"),
            ("#0ea5a4", "Avg SLA (hrs)", f"{avg_sla_hours:.1f}", "Avg SLA"),
            ("#0f172a", "Avg Job Value", f"${avg_value:,.0f}", "Average estimate"),
        ]

        st.markdown("<div style='display:flex;flex-wrap:wrap;'>", unsafe_allow_html=True)
        # Render two rows of 4 columns
        for idx, (col_color, label, value, note) in enumerate(KPI_COLORS):
            # Use inline block style to make 4 columns per row (approx)
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {col_color}, {col_color}); width:24%; margin-right:1%; color: #ffffff; display:inline-block;">
                    <div style="font-size:12px; font-weight:600; opacity:0.9;">{label}</div>
                    <div style="font-size:28px; font-weight:800; margin-top:8px; color:#ffffff;">{value}</div>
                    <div style="margin-top:6px; color: rgba(255,255,255,0.9); font-size:12px;">{note}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # Stage breakdown cards (2 rows x 4 columns)
        st.markdown("### 📈 Pipeline Stages")
        stage_colors = {
            LeadStatus.NEW: "#2563eb",
            LeadStatus.CONTACTED: "#eab308",
            LeadStatus.INSPECTION_SCHEDULED: "#f97316",
            LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
            LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
            LeadStatus.AWARDED: "#22c55e",
            LeadStatus.LOST: "#ef4444"
        }
        stage_counts = df["status"].value_counts().to_dict()

        # Layout two rows of up to 4 items each
        statuses = LeadStatus.ALL.copy()
        row1 = statuses[:4]
        row2 = statuses[4:8]

        def render_stage_row(row_statuses):
            cols = st.columns(len(row_statuses))
            for i, status in enumerate(row_statuses):
                cnt = int(stage_counts.get(status, 0))
                pct = (cnt / total_leads * 100) if total_leads else 0
                color = stage_colors.get(status, "#000000")
                with cols[i]:
                    st.markdown(f"""
                    <div class="stage-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-weight:800; font-size:14px;">{status}</div>
                            <div style="font-weight:700; font-size:20px; color:{color};">{cnt}</div>
                        </div>
                        <div class="progress-bar"><div class="progress-fill" style="width:{pct}%; background:{color};"></div></div>
                        <div style="text-align:center; margin-top:6px; color:var(--muted); font-size:12px;">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        render_stage_row(row1)
        render_stage_row(row2)

        st.markdown("---")

        # ---------- Priority Leads (Top 8) ----------
        st.markdown("### 🎯 Priority Leads (Top 8)")
        priority_list = []
        for _, row in df.iterrows():
            try:
                score, *_ = compute_priority_for_lead_row(row, weights)
            except Exception:
                score = 0.0
            # SLA calculation
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            elif pd.isna(sla_entered):
                sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0

            # predicted conversion
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
                "time_left_hours": float(remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "conversion_prob": prob,
                "damage_type": row.get("damage_type", "Unknown")
            })

        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                status = r["status"]
                status_color = stage_colors.get(status, "#ffffff")

                # Priority badge color & label
                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"
                    priority_label = "🟢 NORMAL"

                # SLA status (time-left red)
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r['time_left_hours'])
                    mins_left = int((r['time_left_hours'] * 60) % 60)
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h {mins_left}m left</span>"

                # Conversion probability
                conv_html = ""
                if r["conversion_prob"] is not None:
                    conv_pct = r["conversion_prob"] * 100
                    conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                    conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"

                # Render priority card — ensure HTML tags are balanced
                st.markdown(f"""
                <div style="background: rgba(0,0,0,0.03); padding:12px; border-radius:10px; margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                      <div style="margin-bottom:8px;">
                        <span style="color:{priority_color};font-weight:800;font-size:14px;">{priority_label}</span>
                        <span style="display:inline-block; padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-left:8px; background:{status_color}22; color:{status_color}; border:1px solid {status_color}44;">{status}</span>
                      </div>
                      <div style="font-size:16px; font-weight:700; color:var(--text); margin-bottom:4px;">
                        #{int(r['id'])} — {r['contact_name']}
                      </div>
                      <div style="font-size:13px; color:var(--muted); margin-bottom:8px;">
                        {r['damage_type'].title()} | Est: <span style="color:var(--money-green); font-weight:800;">${r['estimated_value']:,.0f}</span>
                      </div>
                      <div style="font-size:13px;">
                        {sla_html}
                        {conv_html}
                      </div>
                    </div>
                    <div style="text-align:right; padding-left:20px;">
                      <div style="font-size:28px; font-weight:800; color:{priority_color};">{score:.2f}</div>
                      <div style="font-size:11px; color:var(--muted); text-transform:uppercase;">Priority</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No priority leads to display.")

        st.markdown("---")

        # ---------- Expandable All Leads (edit) ----------
        st.markdown("### 📋 All Leads (expand a card to edit / change status)")
        for lead in leads:
            est_val_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at.strftime('%Y-%m-%d %H:%M') if lead.created_at else '—'}")

                with colB:
                    entered = lead.sla_entered_at or lead.created_at
                    if isinstance(entered, str):
                        try:
                            entered = datetime.fromisoformat(entered)
                        except:
                            entered = datetime.utcnow()
                    if entered is None:
                        entered = datetime.utcnow()
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    if remaining.total_seconds() <= 0:
                        sla_status_html = "<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        sla_status_html = f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours}h {mins}m</div>"
                    st.markdown(f"""
                        <div style='text-align:right;'>
                            <div style='display:inline-block; padding:6px 12px; border-radius:20px; background:{stage_colors.get(lead.status,'#000000')}22; color:{stage_colors.get(lead.status,'#000000')}; font-weight:700;'>{lead.status}</div>
                            <div style='margin-top:12px;'>{sla_status_html}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact buttons
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    with qc1:
                        st.markdown(f"<a href='tel:{phone}'><button class='btn-inline' style='background:var(--call-blue); color:#fff;'>📞 Call</button></a>", unsafe_allow_html=True)
                    with qc2:
                        wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                        wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                        st.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-inline' style='background:var(--wa-green); color:#000;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")

                if email:
                    with qc3:
                        st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button class='btn-inline' style='background:transparent; color:var(--text); border:1px solid #e5e7eb;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")

                qc4.write("")

                st.markdown("---")

                # Lead edit form
                with st.form(f"update_lead_{lead.id}"):
                    st.markdown("#### Update Lead")
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"status_{lead.id}")
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        new_contacted = st.checkbox("Contacted", value=bool(lead.contacted), key=f"contacted_{lead.id}")
                    with ucol2:
                        insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.inspection_scheduled), key=f"insp_sched_{lead.id}")
                        insp_comp = st.checkbox("Inspection Completed", value=bool(lead.inspection_completed), key=f"insp_comp_{lead.id}")
                        est_sub = st.checkbox("Estimate Submitted", value=bool(lead.estimate_submitted), key=f"est_sub_{lead.id}")

                    new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                    new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead.estimated_value or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead.id}")

                    # If user sets AWARDED, show invoice uploader and comment field
                    awarded_invoice_file = None
                    awarded_comment = ""
                    lost_comment = ""
                    if new_status == LeadStatus.AWARDED:
                        st.markdown("**Award details**")
                        awarded_comment = st.text_area("Award comment", key=f"award_comment_{lead.id}")
                        awarded_invoice_file = st.file_uploader("Upload Invoice File (optional) — only for Awarded", type=["pdf","jpg","jpeg","png","xlsx","csv"], key=f"award_inv_{lead.id}")
                    elif new_status == LeadStatus.LOST:
                        st.markdown("**Lost details**")
                        lost_comment = st.text_area("Lost comment", key=f"lost_comment_{lead.id}")

                    if st.form_submit_button("💾 Update Lead"):
                        try:
                            # update DB record safely
                            db_lead = s.query(Lead).filter(Lead.id == lead.id).first()
                            if db_lead:
                                db_lead.status = new_status
                                db_lead.assigned_to = new_assigned
                                db_lead.contacted = bool(new_contacted)
                                db_lead.inspection_scheduled = bool(insp_sched)
                                db_lead.inspection_completed = bool(insp_comp)
                                db_lead.estimate_submitted = bool(est_sub)
                                db_lead.notes = new_notes
                                db_lead.estimated_value = float(new_est_val or 0.0)

                                # SLA entered when status first set? keep existing sla_entered_at if present
                                if db_lead.sla_entered_at is None:
                                    db_lead.sla_entered_at = datetime.utcnow()

                                if new_status == LeadStatus.AWARDED:
                                    db_lead.awarded_date = datetime.utcnow()
                                    db_lead.awarded_comment = awarded_comment
                                    if awarded_invoice_file is not None:
                                        path = save_uploaded_file(awarded_invoice_file, prefix=f"lead_{db_lead.id}_inv")
                                        db_lead.awarded_invoice = path
                                if new_status == LeadStatus.LOST:
                                    db_lead.lost_date = datetime.utcnow()
                                    db_lead.lost_comment = lost_comment

                                s.add(db_lead)
                                s.commit()
                                st.success(f"Lead #{db_lead.id} updated.")
                            else:
                                st.error("Lead not found (was it deleted?).")
                        except Exception as e:
                            st.error(f"Failed to update lead: {str(e)}")
                            st.write(traceback.format_exc())

        # End of 'All leads' loop

# ---------------------------
# Page: Analytics & SLA (donut pie)
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard (Donut chart)")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Build status counts for pie/donut
        try:
            status_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        except Exception:
            # fallback: groupby safe
            status_counts = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0)

        pie_df = pd.DataFrame({
            "status": status_counts.index,
            "count": status_counts.values
        })

        colors = [ "#2563eb", "#eab308", "#f97316", "#14b8a6", "#a855f7", "#22c55e", "#ef4444" ]
        fig = px.pie(
            pie_df, names="status", values="count", hole=0.45,
            color="status", color_discrete_sequence=colors[:len(pie_df)]
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(orientation="h", y=-0.1), margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # SLA overdue table
        st.subheader("SLA / Overdue Leads")
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
            overdue = remaining.total_seconds() <= 0
            overdue_rows.append({
                "id": row.get("id"),
                "contact": row.get("contact_name"),
                "status": row.get("status"),
                "deadline": deadline,
                "overdue": overdue
            })
        df_overdue = pd.DataFrame(overdue_rows)
        if not df_overdue.empty:
            st.dataframe(df_overdue.sort_values("deadline"))
        else:
            st.info("No SLA overdue leads.")

# ---------------------------
# Page: Exports
# ---------------------------
elif page == "Exports":
    st.header("📤 Export data")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads yet to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    df_est = estimates_df(s)
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")

# ---------------------------
# END
# ---------------------------
