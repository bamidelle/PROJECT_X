"""
project_x_pipeline_glassy.py
Single-file Streamlit app (Option A) — Redesigned Pipeline Dashboard (Glassy dark/grey background),
animated responsive cards, left-aligned big colored numbers, white medium titles, status labels,
award/upload invoice handling, priority leads, SLA, donut analytics (auto-updating), and exports.

Notes:
- Uses SQLite + SQLAlchemy for storage (file: project_x_pipeline.db by default)
- No st.experimental_rerun() calls (avoids AttributeError in some deployments)
- If plotly is not installed, the Analytics page will fall back to a simple table and a message.
- Save this file into your app folder and run with `streamlit run project_x_pipeline_glassy.py`
"""

import os
from datetime import datetime, timedelta
import traceback

import streamlit as st
import pandas as pd

# Optional plotting library
try:
    import plotly.express as px
except Exception:
    px = None

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# ---------------------------
# CONFIG / DB
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_pipeline.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


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
        LOST,
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
# DB helpers
# ---------------------------
def init_db():
    Base.metadata.create_all(bind=engine)
    # migration-safe: ensure columns exist (lightweight)
    inspector = inspect(engine)
    if "leads" in inspector.get_table_names():
        existing = {c["name"] for c in inspector.get_columns("leads")}
        extras = {
            "awarded_invoice": "TEXT",
        }
        conn = engine.connect()
        for col, typ in extras.items():
            if col not in existing:
                try:
                    conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {typ};")
                except Exception:
                    pass
        conn.close()


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
        qualified=bool(kwargs.get("qualified", False)),
        estimated_value=float(kwargs.get("estimated_value")) if kwargs.get("estimated_value") not in (None, "") else None,
        sla_entered_at=datetime.utcnow()
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead


def create_estimate(session, lead_id, amount, details=""):
    est = Estimate(lead_id=lead_id, amount=float(amount), details=details)
    session.add(est)
    session.commit()
    session.refresh(est)
    # mark lead
    lead = session.query(Lead).filter(Lead.id == lead_id).first()
    if lead:
        lead.estimate_submitted = True
        lead.estimate_submitted_at = datetime.utcnow()
        lead.status = LeadStatus.ESTIMATE_SUBMITTED
        session.add(lead)
        session.commit()
    return est


def leads_df(session):
    rows = session.query(Lead).order_by(Lead.created_at.desc()).all()
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
    # ensure columns exist even when empty
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


# ---------------------------
# Priority scoring & SLA helpers
# ---------------------------
def compute_priority_for_lead_row(lead_row, weights):
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
    except Exception:
        val = 0.0
    baseline = weights.get("value_baseline", 5000.0)
    value_score = min(val / max(1.0, baseline), 1.0)

    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    try:
        if sla_entered is None:
            time_left_h = 9999.0
        else:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            deadline = sla_entered + timedelta(hours=int(lead_row.get("sla_hours") or 24))
            time_left_h = max((deadline - datetime.utcnow()).total_seconds() / 3600.0, 0.0)
    except Exception:
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


# ---------------------------
# UI CSS (Glassy grey/black background + animated cards)
# ---------------------------
GLASSY_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&family=Comfortaa:wght@700&display=swap');

:root{
  --bg-grad: linear-gradient(180deg, #0f1720 0%, #1b2330 100%);
  --card-bg: rgba(20,23,26,0.55);
  --card-blur: 8px;
  --muted: #9aa4b2;
  --white: #ffffff;
  --accent: #2563eb;
  --money: #22c55e;
  --glass-border: rgba(255,255,255,0.06);
}

/* Page background */
body, .stApp {
  background: var(--bg-grad);
  color: var(--muted);
  font-family: 'Poppins', sans-serif;
}

/* header */
.header {
  color: var(--white);
  font-family: 'Comfortaa', cursive;
  font-weight:700;
  font-size:20px;
  padding:10px 0;
}

/* KPI card */
.kpi-card {
  border-radius:12px;
  padding:18px;
  margin:8px;
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border:1px solid var(--glass-border);
  backdrop-filter: blur(var(--card-blur));
  transition: transform .22s ease, box-shadow .22s ease;
  box-shadow: 0 6px 18px rgba(2,6,23,0.35);
}
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 14px 40px rgba(2,6,23,0.45); }

/* big left-aligned number */
.kpi-number {
  font-size:34px;
  font-weight:800;
  text-align:left;
  margin:0;
  padding:0;
}

/* kpi label */
.kpi-label {
  color: var(--white);
  font-size:14px;
  font-weight:600;
  margin-bottom:6px;
  text-align:left;
}

/* small note under kpi */
.kpi-note { color:var(--muted); font-size:12px; margin-top:8px; }

/* Stage card */
.stage-card {
  background: rgba(0,0,0,0.6);
  color: #fff;
  padding:14px;
  border-radius:10px;
  border:1px solid rgba(255,255,255,0.04);
  transition: transform .18s ease;
}
.stage-card:hover { transform: translateY(-4px); }

/* progress bar */
.stage-bar { width:100%; height:10px; background: rgba(255,255,255,0.06); border-radius:6px; overflow:hidden; }
.stage-fill { height:100%; border-radius:6px; transition: width .6s cubic-bezier(.2,.9,.2,1); }

/* status pill */
.status-pill { display:inline-block; padding:6px 12px; border-radius:20px; font-weight:700; font-size:12px; }

/* priority card */
.priority-card {
  background: rgba(255,255,255,0.02);
  border-radius:12px;
  padding:12px;
  margin:8px 0;
  border:1px solid rgba(255,255,255,0.03);
  transition: transform .18s ease;
}
.priority-card:hover { transform: translateY(-4px); }

/* animated medium-length button */
.btn-animated {
  display:inline-block; padding:10px 18px; border-radius:10px; font-weight:700; border:none; cursor:pointer;
  transition: transform .14s ease, box-shadow .14s ease;
}
.btn-animated:hover { transform: translateY(-3px); box-shadow:0 8px 20px rgba(0,0,0,0.25); }

/* small muted */
.small-muted { color: var(--muted); font-size:12px; }

/* ensure expander content text is white for titles */
[data-testid="stExpander"] .css-1l02v3k { color: var(--white) !important; }
"""

# ---------------------------
# APP START
# ---------------------------
st.set_page_config(page_title="Project X — Glassy Pipeline", layout="wide", initial_sidebar_state="expanded")
init_db()
st.markdown(f"<style>{GLASSY_CSS}</style>", unsafe_allow_html=True)
st.markdown('<div class="header">Project X — Pipeline (Glassy)</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=1)
    st.markdown("---")
    st.subheader("Priority weights")
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }
    st.session_state.weights["value_weight"] = st.slider("Value weight", 0.0, 1.0, float(st.session_state.weights["value_weight"]), step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA weight", 0.0, 1.0, float(st.session_state.weights["sla_weight"]), step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Urgency weight", 0.0, 1.0, float(st.session_state.weights["urgency_weight"]), step=0.05)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", contact_name="Demo Customer", contact_phone="+15550000",
                 contact_email="demo@example.com", property_address="100 Demo Ave", damage_type="water",
                 assigned_to="Alex", estimated_value=4500, notes="Demo lead", sla_hours=24, qualified=True)
        st.success("Demo lead added")

# little auto-reload helper via HTML if desired (safe)
if "pipeline_autorefresh" not in st.session_state:
    st.session_state.pipeline_autorefresh = True
if st.session_state.pipeline_autorefresh:
    st.components.v1.html(
        "<script>setTimeout(()=>{ window.location.reload(); }, 30000);</script>",
        height=0
    )

# Stage color map
STAGE_COLORS = {
    LeadStatus.NEW: "#2563eb",
    LeadStatus.CONTACTED: "#eab308",
    LeadStatus.INSPECTION_SCHEDULED: "#f97316",
    LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
    LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
    LeadStatus.AWARDED: "#22c55e",
    LeadStatus.LOST: "#ef4444",
}

# ---------------------------
# Page: Leads / Capture
# ---------------------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture", anchor="lead_capture")
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
            qualified_choice = st.selectbox("Is the Lead Qualified?", ["Yes", "No"], index=0)
            sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=24, step=1)
        notes = st.text_area("Notes", placeholder="Additional context.")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        submitted = st.form_submit_button("Create Lead", help="Create a new lead")
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
    s = get_session()
    df = leads_df(s)
    st.subheader("Recent leads")
    if df.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# ---------------------------
# Page: Pipeline Board (Glassy redesign)
# ---------------------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Glassy", anchor="pipeline")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # KPI cards arranged 2 rows x 4 columns visually (wrap)
        total_leads = len(df)
        qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
        total_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
        awarded_leads = int(df[df["status"] == LeadStatus.AWARDED].shape[0]) if not df.empty else 0
        lost_leads = int(df[df["status"] == LeadStatus.LOST].shape[0]) if not df.empty else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads
        avg_value = (total_value / total_leads) if total_leads else 0.0

        KPI_DEFS = [
            {"color": "#1e3a8a", "label": "Total Leads", "value": str(total_leads)},
            {"color": "#7c3aed", "label": "Pipeline Value", "value": f"${total_value:,.0f}"},
            {"color": "#16a34a", "label": "Conversion Rate", "value": f"{conversion_rate:.1f}%"},
            {"color": "#ea580c", "label": "Active Leads", "value": str(active_leads)},
            {"color": "#ef4444", "label": "Awarded", "value": str(awarded_leads)},
            {"color": "#db2777", "label": "Lost", "value": str(lost_leads)},
            {"color": "#0ea5a4", "label": "Avg SLA (hrs)", "value": f"{df['sla_hours'].mean():.1f}" if total_leads else "—"},
            {"color": "#0f172a", "label": "Avg Job Value", "value": f"${avg_value:,.0f}"}
        ]

        # Render KPI cards (flex/wrap)
        st.markdown("<div style='display:flex;flex-wrap:wrap;'>", unsafe_allow_html=True)
        for k in KPI_DEFS:
            # Big left-aligned colored number, title top-left white
            st.markdown(f"""
                <div class="kpi-card" style="width:23%; margin-right:1%; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
                    <div class="kpi-label">{k['label']}</div>
                    <div class="kpi-number" style="color:{k['color']}; font-family: 'Poppins', sans-serif;">{k['value']}</div>
                    <div class="kpi-note">Live</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # Pipeline stages cards (2 rows)
        st.markdown("### 📈 Pipeline Stages")
        statuses = LeadStatus.ALL.copy()
        # Create two rows of up to 4
        row1 = statuses[:4]
        row2 = statuses[4:8]

        def render_stage_row(row_statuses):
            cols = st.columns(len(row_statuses))
            for i, status in enumerate(row_statuses):
                cnt = int(df[df["status"] == status].shape[0]) if not df.empty else 0
                pct = (cnt / total_leads * 100) if total_leads else 0
                color = STAGE_COLORS.get(status, "#888888")
                with cols[i]:
                    st.markdown(f"""
                        <div class="stage-card">
                          <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="font-weight:700; font-size:14px; color:var(--white);">{status.replace('_',' ').title()}</div>
                            <div style="font-weight:900; font-size:28px; color:{color}; text-align:left;">{cnt}</div>
                          </div>
                          <div style="height:8px;margin-top:10px;" class="stage-bar">
                            <div class="stage-fill" style="width:{pct}%; background:{color};"></div>
                          </div>
                          <div style="margin-top:8px; text-align:center; color:var(--muted); font-size:12px;">{pct:.1f}% of pipeline</div>
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
                score, *_ = compute_priority_for_lead_row(row, weights)
            except Exception:
                score = 0.0
            # SLA calc
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            if pd.isna(sla_entered) or sla_entered is None:
                sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            time_left_h = max(remaining.total_seconds() / 3600.0, 0.0)
            overdue = remaining.total_seconds() <= 0
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left_h),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "damage_type": row.get("damage_type", "Unknown")
            })

        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        if pr_df.empty:
            st.info("No priority leads.")
        else:
            # Render as two columns if screen wide else stack
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                status = r["status"]
                status_color = STAGE_COLORS.get(status, "#ffffff")
                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"
                    priority_label = "🟢 NORMAL"

                # SLA text — time left shown in red
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:800;'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r["time_left_hours"])
                    mins_left = int((r["time_left_hours"] * 60) % 60)
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>⏳ {hours_left}h {mins_left}m left</span>"

                st.markdown(f"""
                    <div class="priority-card">
                      <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div style="flex:1;">
                          <div style="margin-bottom:8px;">
                            <span style="color:{priority_color}; font-weight:800; font-size:13px;">{priority_label}</span>
                            <span style="display:inline-block; padding:6px 12px; border-radius:20px; font-weight:700; margin-left:8px; background:{status_color}22; color:{status_color}; border:1px solid {status_color}44;">{status}</span>
                          </div>
                          <div style="font-size:16px; font-weight:800; color:var(--white);">#{int(r['id'])} — {r['contact_name']}</div>
                          <div style="font-size:13px; color:var(--muted); margin-top:6px;">{r['damage_type'].title()} | Est: <span style="color:var(--money); font-weight:800;">${r['estimated_value']:,.0f}</span></div>
                          <div style="margin-top:8px; font-size:13px;">{sla_html}</div>
                        </div>
                        <div style="text-align:right; padding-left:16px;">
                          <div style="font-size:28px; font-weight:900; color:{priority_color};">{score:.2f}</div>
                          <div style="font-size:11px; color:var(--muted); text-transform:uppercase;">Priority</div>
                        </div>
                      </div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # All Leads Expandable — allow editing, award/lost with invoice upload, create estimate named "Job Value Estimate (USD)"
        st.markdown("### 📋 All Leads (expand a card to edit)")
        for lead in leads:
            est_val_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                colA, colB = st.columns([3, 1])
                with colA:
                    st.markdown(f"**Source:** {lead.source or '—'}   •   **Assigned:** {lead.assigned_to or '—'}")
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
                        sla_status_html = "<div style='color:#ef4444;font-weight:800;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        sla_status_html = f"<div style='color:#ef4444;font-weight:700;'>⏳ {hours}h {mins}m</div>"
                    pill_color = STAGE_COLORS.get(lead.status, "#6b7280")
                    # status display: green for awarded, red for lost
                    st.markdown(f"""
                        <div style='text-align:right;'>
                            <div class="status-pill" style="background:{pill_color}22; color:{pill_color}; border:1px solid {pill_color}44;">{lead.status}</div>
                            <div style="margin-top:12px;">{sla_status_html}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    with qc1:
                        st.markdown(f"<a href='tel:{phone}'><button class='btn-animated' style='background:#2563eb;color:#fff;'>📞 Call</button></a>", unsafe_allow_html=True)
                    with qc2:
                        wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                        wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                        st.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-animated' style='background:#25D366;color:#000;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")

                if email:
                    with qc3:
                        st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button class='btn-animated' style='background:transparent;color:var(--white);border:1px solid rgba(255,255,255,0.06);'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")

                qc4.write("")

                st.markdown("---")

                # Update form
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

                    # Awarded -> invoice uploader
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

                                # set sla_entered_at if not set
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
                            st.error("Failed to update lead.")
                            st.text(traceback.format_exc())

                # Create estimate form (Job Value Estimate)
                with st.form(f"create_est_{lead.id}"):
                    st.markdown("**Job Value Estimate (USD)**")
                    est_amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                    est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                    if st.form_submit_button("➕ Create Estimate"):
                        try:
                            create_estimate(s, lead.id, est_amount, est_details)
                            st.success("Estimate created.")
                        except Exception as e:
                            st.error("Failed to create estimate.")
                            st.text(traceback.format_exc())

# ---------------------------
# Page: Analytics & SLA
# ---------------------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # pie/donut chart of statuses
        try:
            status_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        except Exception:
            status_counts = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0)

        pie_df = pd.DataFrame({
            "status": status_counts.index,
            "count": status_counts.values
        })

        colors = [STAGE_COLORS.get(s, "#888") for s in pie_df["status"]]
        if px is not None:
            fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status", color_discrete_sequence=colors)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Plotly is not installed in this environment. Install plotly to see interactive charts.")
            st.dataframe(pie_df)

        # SLA table (auto-updating)
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
    st.header("📤 Exports")
    s = get_session()
    df_leads = leads_df(s)
    if df_leads.empty:
        st.info("No leads to export.")
    else:
        csv = df_leads.to_csv(index=False).encode("utf-8")
        st.download_button("Download leads.csv", csv, file_name="leads.csv", mime="text/csv")
    df_est = estimates_df(s)
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")

# ---------------------------
# End
# ---------------------------
