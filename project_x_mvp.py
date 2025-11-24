# assan_project_x_singlefile.py
"""
Project X — Single-file Streamlit app (Option A)
- Combines models, utilities, and UI in one file
- White background theme
- Google Ads style KPI cards (non-clickable stage cards)
- Editable pipeline (change status, Awarded/Lost with timestamps)
- Awarded status allows optional invoice file upload (saved to disk)
- "Create New Estimate" renamed to "Job Value Estimate (USD)"
- SLA time-left shown in RED when low/overdue
- Analytics uses a responsive Pie chart
- Guards around joblib import (optional ML model)
- Avoids st.experimental_rerun() to prevent AttributeError
"""

import os
from datetime import datetime, timedelta, time as dtime
import tempfile

import streamlit as st
import pandas as pd
import plotly.express as px

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Optional ML model (safe import)
try:
    import joblib
except Exception:
    joblib = None

# ---------------------------
# CONFIG
# ---------------------------
DB_FILE = os.getenv("PROJECT_X_DB", "assan_project_x.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploaded_invoices")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# MODELS
# ---------------------------
class LeadStatus:
    NEW = "NEW"
    CONTACTED = "CONTACTED"
    INSPECTION_SCHEDULED = "INSPECTION_SCHEDULED"
    INSPECTION_COMPLETED = "INSPECTION_COMPLETED"
    ESTIMATE_SUBMITTED = "ESTIMATE_SUBMITTED"
    AWARDED = "AWARDED"
    LOST = "LOST"

    ALL = [
        NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED,
        ESTIMATE_SUBMITTED, AWARDED, LOST
    ]


class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    source = Column(String, default="")
    source_details = Column(String, default="")
    contact_name = Column(String, default="")
    contact_phone = Column(String, default="")
    contact_email = Column(String, default="")
    property_address = Column(String, default="")
    damage_type = Column(String, default="")
    assigned_to = Column(String, default="")
    notes = Column(Text, default="")
    estimated_value = Column(Float, nullable=True)   # optional pipeline estimate
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default=LeadStatus.NEW)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    awarded_comment = Column(Text, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    lost_comment = Column(Text, nullable=True)
    qualified = Column(Boolean, default=False)


class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    amount = Column(Float, default=0.0)
    details = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)

    lead = relationship("Lead", backref="estimates")


def init_db():
    Base.metadata.create_all(bind=engine)


# ---------------------------
# UTILITIES
# ---------------------------
def get_session():
    return SessionLocal()


def add_lead(
    s,
    source="",
    source_details="",
    contact_name="",
    contact_phone="",
    contact_email="",
    property_address="",
    damage_type="",
    assigned_to="",
    notes="",
    sla_hours=24,
    qualified=False,
    estimated_value=None
):
    lead = Lead(
        source=source,
        source_details=source_details,
        contact_name=contact_name,
        contact_phone=contact_phone,
        contact_email=contact_email,
        property_address=property_address,
        damage_type=damage_type,
        assigned_to=assigned_to,
        notes=notes,
        sla_hours=int(sla_hours) if sla_hours else 24,
        qualified=bool(qualified),
        estimated_value=float(estimated_value) if estimated_value else None,
        created_at=datetime.utcnow()
    )
    s.add(lead)
    s.commit()
    s.refresh(lead)
    return lead


def leads_df(s):
    rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not rows:
        return pd.DataFrame()
    def to_row(l):
        return {
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
            "estimated_value": l.estimated_value or 0.0,
            "sla_hours": l.sla_hours,
            "sla_entered_at": l.sla_entered_at,
            "created_at": l.created_at,
            "status": l.status,
            "contacted": l.contacted,
            "inspection_scheduled": l.inspection_scheduled,
            "inspection_scheduled_at": l.inspection_scheduled_at,
            "inspection_completed": l.inspection_completed,
            "inspection_completed_at": l.inspection_completed_at,
            "estimate_submitted": l.estimate_submitted,
            "estimate_submitted_at": l.estimate_submitted_at,
            "awarded_date": l.awarded_date,
            "awarded_invoice": l.awarded_invoice,
            "awarded_comment": l.awarded_comment,
            "lost_date": l.lost_date,
            "lost_comment": l.lost_comment,
            "qualified": l.qualified
        }
    df = pd.DataFrame([to_row(r) for r in rows])
    return df


def create_estimate(s, lead_id, amount, details):
    est = Estimate(lead_id=lead_id, amount=float(amount), details=details, created_at=datetime.utcnow())
    s.add(est)
    # mark lead estimate_submitted
    lead = s.query(Lead).get(lead_id)
    if lead:
        lead.estimate_submitted = True
        lead.estimate_submitted_at = datetime.utcnow()
    s.commit()
    return est


def save_uploaded_file(uploaded_file, lead_id):
    if uploaded_file is None:
        return None
    filename = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def compute_priority_for_lead_row(lead_row, weights):
    """
    Simple priority calculation:
      - value_score: normalized estimated_value vs baseline
      - sla_score: urgency based on hours left
      - urgency flag score: not-contacted / not-scheduled / no-estimate
    Returns (score, value_score, sla_score, contacted_flag, inspection_flag, estimate_flag, time_left_hours)
    """
    try:
        val = float(lead_row.get("estimated_value") or 0.0)
    except Exception:
        val = 0.0
    baseline = float(weights.get("value_baseline", 5000.0))
    value_score = min(val / max(baseline, 1.0), 1.0)

    # compute time left
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
# UI / STREAMLIT
# ---------------------------
init_db()

# White background CSS
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
:root{
  --bg:#ffffff;
  --muted:#6b7280;
  --white:#0b0f13;
  --placeholder:#9ca3af;
  --radius:10px;
  --primary-red:#ef4444;
  --money-green:#16a34a;
  --call-blue:#2563eb;
  --wa-green:#25D366;
}
body, .stApp {
  background: var(--bg);
  color: var(--white);
  font-family: 'Roboto', sans-serif;
}
section[data-testid="stSidebar"] { background: #f8fafc !important; padding: 18px; border-right: 1px solid #e6e9ee;}
.header { padding: 12px; color: var(--white); font-weight:700; font-size:18px; }
.metric-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.98));
    border-radius: 12px; padding: 16px; margin: 8px 0;
    border: 1px solid rgba(15, 23, 42, 0.04);
}
.stage-badge { padding:6px 12px; border-radius:20px; font-size:12px; font-weight:600; margin:4px; display:inline-block; }
.button-primary { background:#ef4444; color:#fff; border:none; padding:8px 12px; border-radius:8px; font-weight:700; }
.small-muted { color: #6b7280; font-size:12px; }
"""

st.set_page_config(page_title="Assan — CRM", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div style='font-weight:800; font-size:20px; color:#0b0f13;'>Assan — Sales & Conversion Tracker</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"], index=0)
    st.markdown("---")

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
    st.markdown('<small class="small-muted">Tip: Increase SLA weight to prioritise leads nearing deadline; increase value weight to prioritise larger jobs.</small>', unsafe_allow_html=True)

# --- Page: Leads / Capture
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    with st.form("lead_form"):
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
        est_val = st.number_input("Initial Estimated Job Value (USD)", min_value=0.0, value=0.0, step=100.0)
        submitted = st.form_submit_button("Create Lead", help="Creates a new lead")
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
                estimated_value=float(est_val) if est_val else None
            )
            st.success(f"Lead created (ID: {lead.id})")

    st.markdown("---")
    st.subheader("Recent leads")
    s = get_session()
    df_recent = leads_df(s)
    if df_recent.empty:
        st.info("No leads yet. Create one above.")
    else:
        st.dataframe(df_recent.sort_values("created_at", ascending=False).head(50))

# --- Page: Pipeline Board
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # Try load model if available (no crash if joblib missing)
        lead_model = None
        try:
            if joblib is not None:
                model_path = os.path.join(os.getcwd(), "lead_conversion_model.pkl")
                if os.path.exists(model_path):
                    lead_model = joblib.load(model_path)
        except Exception:
            lead_model = None

        # ==================== GOOGLE ADS-STYLE CARDS ====================
        # style already set in CSS above

        # Calculate metrics
        total_leads = len(df)
        qualified_leads = len(df[df['qualified'] == True])
        total_value = df['estimated_value'].sum()
        awarded_leads = len(df[df['status'] == LeadStatus.AWARDED])
        lost_leads = len(df[df['status'] == LeadStatus.LOST])

        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0

        stage_counts = df['status'].value_counts().to_dict()
        stage_colors = {
            LeadStatus.NEW: "#2563eb",
            LeadStatus.CONTACTED: "#eab308",
            LeadStatus.INSPECTION_SCHEDULED: "#f97316",
            LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
            LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
            LeadStatus.AWARDED: "#22c55e",
            LeadStatus.LOST: "#ef4444"
        }

        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#6b7280;font-weight:600;">Total Leads</div>
                <div style="font-size:28px;font-weight:800;color:#2563eb;">{total_leads}</div>
                <div style="font-size:12px;color:#16a34a;font-weight:700;">{qualified_leads} Qualified</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#6b7280;font-weight:600;">Pipeline Value</div>
                <div style="font-size:28px;font-weight:800;color:#16a34a;">${total_value:,.0f}</div>
                <div style="font-size:12px;color:#6b7280;font-weight:700;">Active</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#6b7280;font-weight:600;">Conversion Rate</div>
                <div style="font-size:28px;font-weight:800;color:#7c3aed;">{conversion_rate:.1f}%</div>
                <div style="font-size:12px;color:#6b7280;font-weight:700;">{awarded_leads}/{closed_leads} Won</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            active_leads = total_leads - closed_leads
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:12px;color:#6b7280;font-weight:600;">Active Leads</div>
                <div style="font-size:28px;font-weight:800;color:#f97316;">{active_leads}</div>
                <div style="font-size:12px;color:#6b7280;font-weight:700;">In Progress</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Stage breakdown (non-clickable cards)
        st.markdown("### 📈 Pipeline Stages")
        cols = st.columns(len(LeadStatus.ALL))
        for i, stage in enumerate(LeadStatus.ALL):
            count = stage_counts.get(stage, 0)
            pct = (count / total_leads * 100) if total_leads else 0
            color = stage_colors.get(stage, "#111827")
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:12px;color:#6b7280;font-weight:700;">{stage}</div>
                    <div style="font-size:20px;font-weight:800;color:{color};">{count}</div>
                    <div style="width:100%; height:8px; background:#eef2ff; border-radius:6px; margin-top:8px;">
                        <div style="width:{pct}%; height:100%; background:{color}; border-radius:6px;"></div>
                    </div>
                    <div style="font-size:12px;color:#6b7280; text-align:center; margin-top:6px;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Priority Leads (Top 8) — compute priorities including SLA/time-left and predicted probability (if model exists)
        st.markdown("### 🎯 Priority Leads (Top 8)")
        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left = compute_priority_for_lead_row(row, weights)
            # SLA calculation (robust)
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except Exception:
                    sla_entered = datetime.utcnow()
            if pd.isna(sla_entered) if hasattr(pd, "isna") else False:
                sla_entered = datetime.utcnow()
            try:
                deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            except Exception:
                deadline = datetime.utcnow() + timedelta(hours=24)
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0

            prob = None
            if lead_model is not None:
                try:
                    # Predict using a helper if available in your utils/model - simplified here
                    lead_row = row.to_dict() if hasattr(row, "to_dict") else dict(row)
                    # Example: predict_lead_probability(lead_model, lead_row)
                    # We'll try model.predict_proba if available
                    if hasattr(lead_model, "predict_proba"):
                        # create feature vector safely - this is a placeholder; real model requires feature engineering
                        prob = None
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

        if pr_df.empty:
            st.info("No priority leads to display.")
        else:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                status_color = stage_colors.get(r["status"], "#111827")

                if score >= 0.7:
                    priority_color = "#ef4444"
                    priority_label = "🔴 CRITICAL"
                elif score >= 0.45:
                    priority_color = "#f97316"
                    priority_label = "🟠 HIGH"
                else:
                    priority_color = "#16a34a"
                    priority_label = "🟢 NORMAL"

                # SLA display (time-left red if <= 24h or overdue)
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:#ef4444;font-weight:700;'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r['time_left_hours'])
                    mins_left = int((r['time_left_hours'] * 60) % 60)
                    # red color when under 24 hours
                    time_color = "#ef4444" if r['time_left_hours'] <= 24 else "#2563eb"
                    sla_html = f"<span style='color:{time_color};font-weight:600;'>⏳ {hours_left}h {mins_left}m left</span>"

                conv_html = ""
                if r["conversion_prob"] is not None:
                    conv_pct = r["conversion_prob"] * 100
                    conv_color = "#16a34a" if conv_pct > 70 else ("#f97316" if conv_pct > 40 else "#ef4444")
                    conv_html = f"<span style='color:{conv_color};font-weight:600;margin-left:12px;'>📊 {conv_pct:.0f}% Win Prob</span>"

                st.markdown(f"""
                <div class="metric-card" style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                        <div style="margin-bottom:8px;">
                            <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                            <span class="stage-badge" style="background:{status_color}20; color:{status_color}; border:1px solid {status_color}40;">{r['status']}</span>
                        </div>
                        <div style="font-size:16px; font-weight:800; color:#0b0f13;">#{int(r['id'])} — {r['contact_name']}</div>
                        <div style="font-size:13px; color:#6b7280; margin-top:6px;">
                            {r['damage_type'].title()} | Est: <span style="color:#16a34a; font-weight:800;">${r['estimated_value']:,.0f}</span>
                        </div>
                        <div style="font-size:13px; margin-top:8px;">
                            {sla_html}
                            {conv_html}
                        </div>
                    </div>
                    <div style="text-align:right; padding-left:20px;">
                        <div style="font-size:28px; font-weight:800; color:{priority_color};">{score:.2f}</div>
                        <div style="font-size:11px; color:#6b7280; text-transform:uppercase;">Priority</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed Lead Cards (Editable) - this preserves the previous editable behavior
        st.markdown("### 📋 All Leads (expand a card to edit / change status)")
        for lead in leads:
            est_val = lead.estimated_value or 0.0
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — ${est_val:,.0f}"
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
                        except Exception:
                            entered = datetime.utcnow()
                    deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                    remaining = deadline - datetime.utcnow()
                    if remaining.total_seconds() <= 0:
                        sla_status = f"<div style='color:#ef4444;font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        time_color = "#ef4444" if remaining.total_seconds() <= 24 * 3600 else "#2563eb"
                        sla_status = f"<div style='color:{time_color};font-weight:600;'>⏳ {hours}h {mins}m</div>"
                    st.markdown(f"<div style='text-align:right;'>{sla_status}</div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact buttons
                qc1, qc2, qc3, qc4 = st.columns([1, 1, 1, 4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    qc1.markdown(f"<a href='tel:{phone}'><button style='background:#2563eb;color:#fff;border:none;border-radius:8px;padding:8px 12px;'>📞 Call</button></a>", unsafe_allow_html=True)
                    wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                    wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button style='background:#25D366;color:#fff;border:none;border-radius:8px;padding:8px 12px;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" "); qc2.write(" ")
                if email:
                    qc3.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button style='background:transparent; color:#0b0f13; border:1px solid #e6e9ee; border-radius:8px; padding:8px 12px;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")

                st.markdown("---")

                # Lead update form (editable fields + status change)
                with st.form(f"update_lead_{lead.id}"):
                    st.markdown("#### Update Lead")
                    ucol1, ucol2 = st.columns(2)
                    with ucol1:
                        new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status))
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "")
                        contacted = st.checkbox("Contacted", value=lead.contacted)
                    with ucol2:
                        inspection_scheduled = st.checkbox("Inspection Scheduled", value=lead.inspection_scheduled)
                        inspection_completed = st.checkbox("Inspection Completed", value=lead.inspection_completed)
                        estimate_submitted = st.checkbox("Estimate Submitted", value=lead.estimate_submitted)
                    new_notes = st.text_area("Notes", value=lead.notes or "")

                    # If user selects AWARDED, show invoice upload + comment + awarded date auto-set
                    invoice_file = None
                    awarded_comment = ""
                    lost_comment = ""
                    if new_status == LeadStatus.AWARDED:
                        st.markdown("**Awarded Job Details**")
                        awarded_comment = st.text_area("Awarded comment (optional)", value=lead.awarded_comment or "")
                        invoice_file = st.file_uploader("Upload Invoice (optional, PDF/PNG/JPG)", type=["pdf", "png", "jpg", "jpeg"])
                    if new_status == LeadStatus.LOST:
                        st.markdown("**Lost Job Details**")
                        lost_comment = st.text_area("Lost comment (optional)", value=lead.lost_comment or "")

                    if st.form_submit_button("💾 Update Lead"):
                        # update attributes and commit
                        try:
                            lead.status = new_status
                            lead.assigned_to = new_assigned
                            lead.contacted = bool(contacted)
                            lead.inspection_scheduled = bool(inspection_scheduled)
                            lead.inspection_completed = bool(inspection_completed)
                            lead.estimate_submitted = bool(estimate_submitted)
                            lead.notes = new_notes

                            # status-specific actions
                            if new_status == LeadStatus.AWARDED:
                                lead.awarded_date = datetime.utcnow()
                                lead.awarded_comment = awarded_comment or None
                                if invoice_file:
                                    path = save_uploaded_file(invoice_file, lead.id)
                                    lead.awarded_invoice = path
                                # set estimated_value if not set
                                if lead.estimated_value in (None, 0.0):
                                    # ask user for estimate? keep as-is — optional step omitted here
                                    pass
                            elif new_status == LeadStatus.LOST:
                                lead.lost_date = datetime.utcnow()
                                lead.lost_comment = lost_comment or None

                            # Set SLA entered time if not present
                            if not lead.sla_entered_at:
                                lead.sla_entered_at = datetime.utcnow()

                            s.add(lead)
                            s.commit()
                            st.success(f"Lead #{lead.id} updated.")
                        except Exception as e:
                            st.error(f"Failed to update lead: {e}")

                st.markdown("---")

                # Estimates section
                st.markdown("#### 💰 Job Value Estimate (USD)")
                lead_estimates = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if lead_estimates:
                    for est in lead_estimates:
                        est_status = "✅ Approved" if est.approved else ("❌ Lost" if est.lost else "⏳ Pending")
                        est_color = "#16a34a" if est.approved else ("#ef4444" if est.lost else "#f97316")
                        st.markdown(f"""
                        <div style="padding:8px;border-radius:8px;background:#ffffff;border:1px solid #eef2ff;margin-bottom:8px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <div>
                                    <span style="color:{est_color};font-weight:800;">{est_status}</span>
                                    <span style="margin-left:12px;color:#16a34a;font-weight:800;font-size:18px;">${est.amount:,.0f}</span>
                                </div>
                                <div style="color:#6b7280;font-size:12px;">{est.created_at.strftime('%Y-%m-%d')}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No estimates yet for this lead.")

                # Create Estimate form (renamed)
                with st.form(f"create_estimate_{lead.id}"):
                    st.markdown("**Create Job Value Estimate (USD)**")
                    est_amount = st.number_input("Amount (USD)", min_value=0.0, step=100.0, key=f"est_amt_{lead.id}")
                    est_details = st.text_area("Details", key=f"est_det_{lead.id}")
                    if st.form_submit_button("➕ Create Estimate"):
                        try:
                            create_estimate(s, lead.id, est_amount, est_details)
                            st.success("Estimate created!")
                        except Exception as e:
                            st.error(f"Failed to create estimate: {e}")

# --- Page: Analytics & SLA
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA Dashboard")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze. Add some leads first.")
    else:
        # Pie chart for stage distribution
        funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
        funnel.columns = ["stage", "count"]
        funnel = funnel[funnel["count"] > 0]  # remove zero slices for clarity
        colors_map = {
            LeadStatus.NEW: "#2563eb",
            LeadStatus.CONTACTED: "#eab308",
            LeadStatus.INSPECTION_SCHEDULED: "#f97316",
            LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
            LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
            LeadStatus.AWARDED: "#22c55e",
            LeadStatus.LOST: "#ef4444"
        }
        color_list = [colors_map.get(s, "#111827") for s in funnel["stage"].tolist()]
        fig = px.pie(funnel, names="stage", values="count", title="Leads by Stage", hole=0.35)
        fig.update_traces(marker=dict(colors=color_list, line=dict(color='#ffffff', width=1)))
        fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("SLA / Overdue Leads")
        overdue_rows = []
        for _, row in df.iterrows():
            sla_entered_at = row["sla_entered_at"] or row["created_at"]
            if isinstance(sla_entered_at, str):
                try:
                    sla_entered_at = datetime.fromisoformat(sla_entered_at)
                except Exception:
                    sla_entered_at = datetime.utcnow()
            sla_hours = int(row["sla_hours"]) if not pd.isna(row["sla_hours"]) else 24
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

# --- Page: Exports
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
        "id": e.id,
        "lead_id": e.lead_id,
        "amount": e.amount,
        "details": e.details,
        "created_at": e.created_at,
        "approved": e.approved,
        "lost": e.lost
    } for e in s.query(Estimate).all()])
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")
