# project_x_singlefile.py
"""
Project X — Single-file Streamlit app (Option A)
Features:
- Single-file SQLite + SQLAlchemy Streamlit app
- Google Ads style pipeline dashboard (2 rows × 4 columns KPI cards)
- Pipeline stage cards (black cards with white text)
- Priority Leads (Top 8) cards with SLA countdown (red), conversion prob stub
- Expandable All Leads: editable, can move to AWARDED/LOST; AWARDED allows invoice upload
- Donut analytics auto-refreshes when data changes (uses page reload JS)
- Fonts: Poppins and Comfortaa (loaded via Google fonts)
- Defensive: no st.experimental_rerun(), optional joblib model support
"""

import os
from datetime import datetime, timedelta
import traceback

import streamlit as st
import pandas as pd
import plotly.express as px

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Optional ML dependency: joblib; if absent we continue without model
try:
    import joblib
except Exception:
    joblib = None

# ---------- Configuration ----------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_singlefile.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
MODEL_PATH = "lead_conversion_model.pkl"  # optional

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ---------- Lead statuses ----------
class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"

    ALL = [
        NEW,
        CONTACTED,
        INSPECTION_SCHEDULED,
        INSPECTION_COMPLETED,
        ESTIMATE_SUBMITTED,
        AWARDED,
        LOST
    ]


# ---------- Models ----------
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
    estimated_value = Column(Float, nullable=True, default=0.0)
    status = Column(String, default=LeadStatus.NEW)
    created_at = Column(DateTime, default=datetime.utcnow)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)

    # flags
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)

    # awarded / lost
    awarded_comment = Column(Text, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(Text, nullable=True)

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


# ---------- DB Utilities ----------
def init_db():
    # create tables and run light migration if DB exists but columns missing
    Base.metadata.create_all(bind=engine)
    # simple inspector-based migration (add missing columns) is intentionally minimal here


def get_session():
    return SessionLocal()


def add_lead(session, **kwargs):
    lead = Lead(
        source=kwargs.get("source", "Unknown"),
        source_details=kwargs.get("source_details"),
        contact_name=kwargs.get("contact_name"),
        contact_phone=kwargs.get("contact_phone"),
        contact_email=kwargs.get("contact_email"),
        property_address=kwargs.get("property_address"),
        damage_type=kwargs.get("damage_type"),
        assigned_to=kwargs.get("assigned_to"),
        notes=kwargs.get("notes"),
        sla_hours=int(kwargs.get("sla_hours", 24)),
        sla_entered_at=datetime.utcnow(),
        qualified=bool(kwargs.get("qualified", False)),
        estimated_value=float(kwargs.get("estimated_value") or 0.0)
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead


def create_estimate(session, lead_id, amount, details=""):
    est = Estimate(lead_id=lead_id, amount=float(amount or 0.0), details=details)
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
            "estimated_value": float(r.estimated_value or 0.0),
            "status": r.status,
            "created_at": r.created_at,
            "sla_hours": int(r.sla_hours or 24),
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
    # ensure columns present even if empty
    if df.empty:
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


# ---------- small utilities ----------
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


def safe_parse_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(v)
    except Exception:
        return None


# ---------- priority scoring (same logic, returns score 0..1) ----------
def compute_priority_for_lead_row(lead_row, weights):
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
    except Exception:
        val = 0.0
    baseline = float(weights.get("value_baseline", 5000.0))
    value_score = min(1.0, val / max(1.0, baseline))

    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    if sla_entered is None:
        time_left_h = 9999.0
    else:
        if isinstance(sla_entered, str):
            try:
                sla_entered = datetime.fromisoformat(sla_entered)
            except Exception:
                sla_entered = datetime.utcnow()
        try:
            deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
            time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
        except Exception:
            time_left_h = 9999.0

    sla_score = max(0.0, (72.0 - min(time_left_h, 72.0)) / 72.0)
    contacted_flag = 0.0 if bool(lead_row.get("contacted")) else 1.0
    inspection_flag = 0.0 if bool(lead_row.get("inspection_scheduled")) else 1.0
    estimate_flag = 0.0 if bool(lead_row.get("estimate_submitted")) else 1.0

    urgency_component = (
        contacted_flag * weights.get("contacted_w", 0.6) +
        inspection_flag * weights.get("inspection_w", 0.5) +
        estimate_flag * weights.get("estimate_w", 0.5)
    )

    total_weight = (
        weights.get("value_weight", 0.5) +
        weights.get("sla_weight", 0.35) +
        weights.get("urgency_weight", 0.15)
    )
    if total_weight <= 0:
        total_weight = 1.0

    score = (
        value_score * weights.get("value_weight", 0.5) +
        sla_score * weights.get("sla_weight", 0.35) +
        urgency_component * weights.get("urgency_weight", 0.15)
    ) / total_weight

    score = max(0.0, min(score, 1.0))
    return score


def predict_lead_probability_safe(lead_model, lead_row):
    if lead_model is None:
        return None
    try:
        # example stub — adapt to your model input contract
        X = pd.DataFrame([{
            "estimated_value": lead_row.get("estimated_value") or 0.0,
            "qualified": 1 if lead_row.get("qualified") else 0,
            "sla_hours": lead_row.get("sla_hours") or 24
        }])
        if hasattr(lead_model, "predict_proba"):
            return float(lead_model.predict_proba(X)[:, 1][0])
        else:
            return float(lead_model.predict(X)[0])
    except Exception:
        return None


# ---------- CSS & Fonts (Poppins + Comfortaa) ----------
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Comfortaa:wght@700&display=swap');
:root{
  --bg:#ffffff;
  --muted:#6b7280;
  --text:#0b0f13;
  --radius:10px;
  --primary-red:#ef4444;
  --money-green:#16a34a;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}

/* Base */
body, .stApp {
  background: var(--bg);
  color: var(--text);
  font-family: 'Poppins', sans-serif;
}

/* Use Comfortaa for headings */
h1, h2, h3, .header {
  font-family: 'Comfortaa', cursive;
}

/* KPI metric card */
.metric-card {
  border-radius: 12px;
  padding: 14px;
  margin: 6px;
  color: #ffffff;
}

/* Stage card (black with white text) */
.stage-card {
  background: #000000;
  color: #ffffff;
  padding: 12px;
  border-radius: 8px;
  margin: 6px;
}

/* Priority card */
.priority-card {
  background: rgba(0,0,0,0.03);
  border-radius: 10px;
  padding: 12px;
  margin-bottom: 8px;
}

/* small text */
.small-muted { color: var(--muted); font-size:12px; }

/* SLA time red emphasis */
.sla-red { color: var(--primary-red); font-weight:700; }

/* Buttons inside HTML */
.btn-inline { padding:6px 10px; border-radius:8px; font-weight:600; border:none; cursor:pointer; }
"""

# ---------- App start ----------
st.set_page_config(page_title="Project X — CRM (Single file)", layout="wide")
init_db()
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

# Display which fonts we are using (for the user's request)
st.markdown("<div class='header'>Project X — Sales & Conversion Tracker</div>", unsafe_allow_html=True)
st.caption("Current app fonts: Poppins (body) & Comfortaa (headings).")

# Sidebar: navigation & controls
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

    # weights in session_state
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5,
            "value_baseline": 5000.0
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

# Client-side auto-reload if enabled (keeps analytics/pipeline current)
if st.session_state.get("pipeline_autorefresh", False):
    st.components.v1.html(
        """
        <script>
        // reload page every 30s
        setTimeout(()=>{ window.location.reload(); }, 30000);
        </script>
        """,
        height=0
    )

# Load model (if available)
lead_model = None
if joblib is not None and os.path.exists(MODEL_PATH):
    try:
        lead_model = joblib.load(MODEL_PATH)
    except Exception:
        lead_model = None

# ---------------- Page: Leads / Capture ----------------
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
        estimated_value = st.number_input("Estimated job value (USD)", min_value=0.0, value=0.0, step=100.0)
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

# ---------------- Page: Pipeline Board ----------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Google Ads style")

    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # KPI calculations
        total_leads = len(df)
        qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
        total_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
        awarded_leads = int(df[df["status"] == LeadStatus.AWARDED].shape[0]) if not df.empty else 0
        lost_leads = int(df[df["status"] == LeadStatus.LOST].shape[0]) if not df.empty else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads
        avg_sla_hours = float(df["sla_hours"].mean()) if (not df.empty) else 0.0
        avg_value = (total_value / total_leads) if total_leads else 0.0

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

        # Render KPI cards in 2 rows × 4 columns
        st.markdown("### 📊 Key Performance Indicators")
        # first row
        cols = st.columns(4)
        for i in range(4):
            color, label, value, note = KPI_COLORS[i]
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {color}, {color});">
                    <div style="font-size:12px; font-weight:700; opacity:0.95;">{label}</div>
                    <div style="font-size:28px; font-weight:800; margin-top:8px;">{value}</div>
                    <div style="margin-top:6px; font-size:12px;">{note}</div>
                </div>
                """, unsafe_allow_html=True)
        # second row
        cols = st.columns(4)
        for i in range(4, 8):
            color, label, value, note = KPI_COLORS[i]
            with cols[i - 4]:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {color}, {color});">
                    <div style="font-size:12px; font-weight:700; opacity:0.95;">{label}</div>
                    <div style="font-size:28px; font-weight:800; margin-top:8px;">{value}</div>
                    <div style="margin-top:6px; font-size:12px;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Stage breakdown (two rows)
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

        # Priority Leads (Top 8)
        st.markdown("### 🎯 Priority Leads (Top 8)")
        priority_list = []
        for _, row in df.iterrows():
            try:
                score = compute_priority_for_lead_row(row, weights)
            except Exception:
                score = 0.0

            # SLA calculation robust
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            sla_dt = safe_parse_dt(sla_entered) or row.get("created_at") or datetime.utcnow()
            try:
                deadline = sla_dt + timedelta(hours=int(row.get("sla_hours") or 24))
            except Exception:
                deadline = datetime.utcnow() + timedelta(hours=24)
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0

            prob = None
            if lead_model is not None:
                prob = predict_lead_probability_safe(lead_model, row)

            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": max(0.0, remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "conversion_prob": prob,
                "damage_type": row.get("damage_type", "Unknown")
            })

        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)
        if not pr_df.empty:
            # Render cards stacked (these are not clickable; responsive and fixed)
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                status_color = stage_colors.get(r["status"], "#ffffff")
                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"
                    priority_label = "🟢 NORMAL"

                if r["sla_overdue"]:
                    sla_html = f"<span class='sla-red'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r['time_left_hours'])
                    mins_left = int((r['time_left_hours'] * 60) % 60)
                    sla_html = f"<span class='sla-red'>⏳ {hours_left}h {mins_left}m left</span>"

                conv_html = ""
                if r["conversion_prob"] is not None:
                    conv_pct = r["conversion_prob"] * 100
                    conv_color = "#22c55e" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                    conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"

                # Balanced HTML block
                st.markdown(f"""
                <div class="priority-card">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                      <div style="margin-bottom:8px;">
                        <span style="color:{priority_color};font-weight:800;font-size:14px;">{priority_label}</span>
                        <span style="display:inline-block; padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin-left:8px; background:{status_color}22; color:{status_color}; border:1px solid {status_color}44;">{r['status']}</span>
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

        # All leads expandable, editable (status change to AWARDED triggers invoice upload)
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
                    entered_dt = safe_parse_dt(entered) or lead.created_at or datetime.utcnow()
                    deadline = entered_dt + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    if remaining.total_seconds() <= 0:
                        sla_status_html = "<div class='sla-red'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        sla_status_html = f"<div class='sla-red'>⏳ {hours}h {mins}m</div>"
                    st.markdown(f"""
                        <div style='text-align:right;'>
                            <div style='display:inline-block; padding:6px 12px; border-radius:20px; background:{stage_colors.get(lead.status,'#000000')}22; color:{stage_colors.get(lead.status,'#000000')}; font-weight:700;'>{lead.status}</div>
                            <div style='margin-top:12px;'>{sla_status_html}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact
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

                # Lead update form
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

                                # set sla_entered_at if not set already
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
                                st.error("Lead not found.")
                        except Exception as e:
                            st.error(f"Failed to update lead: {str(e)}")
                            st.write(traceback.format_exc())

# ---------------- Page: Analytics & SLA ----------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA (Donut)")

    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # status counts in fixed order
        try:
            status_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        except Exception:
            status_counts = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0)

        pie_df = pd.DataFrame({"status": status_counts.index, "count": status_counts.values})
        colors = ["#2563eb", "#eab308", "#f97316", "#14b8a6", "#a855f7", "#22c55e", "#ef4444"]
        fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status", color_discrete_sequence=colors[:len(pie_df)])
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(legend=dict(orientation="h", y=-0.12), margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        # SLA overdue list
        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row.get("sla_entered_at") or row.get("created_at")
            sla_dt = safe_parse_dt(sla_entered_at) or row.get("created_at") or datetime.utcnow()
            sla_hours = int(row.get("sla_hours") or 24)
            deadline = sla_dt + timedelta(hours=sla_hours)
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

# ---------------- Page: Exports ----------------
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
