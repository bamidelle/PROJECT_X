# project_x_singlefile.py
# Single-file Streamlit app — redesigned Pipeline Dashboard (white background)
# Uses: streamlit, sqlalchemy, pandas
# Avoids optional imports that caused ModuleNotFoundError (no plotly/matplotlib)

import os
from datetime import datetime, timedelta
import traceback

import streamlit as st
import pandas as pd

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ------------- CONFIG -------------
DB_FILE = os.getenv("PROJECT_X_DB", "project_x_singlefile.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

# ------------- STATUSES -------------
class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION_SCHEDULED = "Inspection Scheduled"
    INSPECTION_COMPLETED = "Inspection Completed"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION_SCHEDULED, INSPECTION_COMPLETED, ESTIMATE_SUBMITTED, AWARDED, LOST]


# ------------- MODELS -------------
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

    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime, nullable=True)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)

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


# ------------- DB helpers -------------
def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()


def add_lead(session, **kwargs):
    lead = Lead(
        source=kwargs.get("source"),
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
        estimated_value=float(kwargs.get("estimated_value")) if kwargs.get("estimated_value") is not None else None,
        qualified=bool(kwargs.get("qualified", False))
    )
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead


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
            "estimated_value": float(r.estimated_value) if r.estimated_value else 0.0,
            "status": r.status,
            "created_at": r.created_at,
            "sla_hours": int(r.sla_hours or 24),
            "sla_entered_at": r.sla_entered_at,
            "contacted": bool(r.contacted),
            "inspection_scheduled": bool(r.inspection_scheduled),
            "inspection_completed": bool(r.inspection_completed),
            "estimate_submitted": bool(r.estimate_submitted),
            "awarded_date": r.awarded_date,
            "awarded_invoice": r.awarded_invoice,
            "lost_date": r.lost_date,
            "qualified": bool(r.qualified),
        })
    df = pd.DataFrame(data)
    if df.empty:
        # create columns so UI code does not break
        df = pd.DataFrame(columns=[
            "id", "source", "source_details", "contact_name", "contact_phone", "contact_email",
            "property_address", "damage_type", "assigned_to", "notes", "estimated_value",
            "status", "created_at", "sla_hours", "sla_entered_at", "contacted",
            "inspection_scheduled", "inspection_completed", "estimate_submitted",
            "awarded_date", "awarded_invoice", "lost_date", "qualified"
        ])
    return df


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


def save_uploaded_file(uploaded_file, prefix="file"):
    if uploaded_file is None:
        return None
    folder = os.path.join(os.getcwd(), "uploads")
    os.makedirs(folder, exist_ok=True)
    fname = f"{prefix}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder, fname)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ------------- PRIORITY scoring (simple) -------------
def compute_priority_for_lead_row(lead_row, weights):
    # returns score 0..1
    val = float(lead_row.get("estimated_value") or 0.0)
    baseline = float(weights.get("value_baseline", 5000.0))
    value_score = min(1.0, val / max(1.0, baseline))

    # SLA-based score
    sla_entered = lead_row.get("sla_entered_at") or lead_row.get("created_at")
    if sla_entered is None:
        time_left_h = 9999.0
    else:
        try:
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
    return score


# ------------- UI / CSS -------------
WHITE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Comfortaa:wght@700&display=swap');

:root{
  --bg: #ffffff;
  --text: #0b1220;
  --muted: #6b7280;
  --glass-bg: rgba(15, 23, 42, 0.85); /* glassy grey-black for pipeline card background */
  --accent-radius: 14px;
  --money-green: #16a34a;
  --primary-blue: #2563eb;
  --primary-orange: #f97316;
  --primary-purple: #7c3aed;
  --primary-red: #ef4444;
}

/* App base */
body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Poppins', system-ui, Arial, sans-serif !important;
}

/* Header */
.header { font-weight:700; font-size:20px; color:var(--text); padding:8px 0; }

/* KPI metric card */
.metric-card {
  border-radius: var(--accent-radius);
  padding: 18px;
  margin: 6px;
  color: #fff;
  display:inline-block;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover { transform: translateY(-6px); box-shadow: 0 10px 30px rgba(16,24,40,0.12); }

/* Pipeline stage card - glassy black */
.stage-card {
  background: var(--glass-bg);
  color: #fff;
  padding: 14px;
  border-radius: 12px;
  margin: 6px 6px;
  transition: transform 0.2s ease;
}
.stage-card:hover { transform: translateY(-4px); }

/* Big colored number left-aligned */
.kpi-number {
  font-weight: 800;
  font-size: 30px;
  line-height: 1;
  letter-spacing: -0.5px;
  color: #fff;
}

/* small white title */
.kpi-label {
  font-size: 13px;
  color: #fff;
  margin-bottom: 6px;
  display:block;
}

/* pipeline progress bar */
.progress-bar { width:100%; height:10px; background: rgba(255,255,255,0.06); border-radius:8px; overflow:hidden; margin-top:10px; }
.progress-fill { height:100%; border-radius:8px; transition: width 0.6s cubic-bezier(.2,.9,.2,1); }

/* animated buttons (medium length) */
.btn-animated {
  display:inline-block; padding:10px 18px; border-radius:10px; font-weight:700; cursor:pointer; border:none;
  transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.btn-animated:active { transform: translateY(1px); }
"""

# ------------- APP START -------------
st.set_page_config(page_title="Project X — Pipeline", layout="wide")
init_db()
st.markdown(f"<style>{WHITE_CSS}</style>", unsafe_allow_html=True)
st.markdown("<div class='header'>Project X — CRM & Pipeline</div>", unsafe_allow_html=True)

# Sidebar: controls & weight tuning
with st.sidebar:
    st.header("Controls")
    page = st.radio("Go to", ["Leads / Capture", "Pipeline Board", "Analytics & SLA", "Exports"])
    st.markdown("---")
    if "weights" not in st.session_state:
        st.session_state.weights = {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        }
    st.markdown("### Priority weights")
    st.session_state.weights["value_weight"] = st.slider("Value weight", 0.0, 1.0, st.session_state.weights["value_weight"], step=0.05)
    st.session_state.weights["sla_weight"] = st.slider("SLA weight", 0.0, 1.0, st.session_state.weights["sla_weight"], step=0.05)
    st.session_state.weights["urgency_weight"] = st.slider("Urgency weight", 0.0, 1.0, st.session_state.weights["urgency_weight"], step=0.05)
    st.markdown("---")
    if st.button("Add Demo Lead"):
        s = get_session()
        add_lead(s,
                 source="Google Ads", contact_name="Demo Customer", contact_phone="+1555000",
                 contact_email="demo@example.com", property_address="100 Demo Ave",
                 damage_type="water", assigned_to="Alex", estimated_value=4500, notes="Demo lead", sla_hours=24, qualified=True)
        st.success("Demo lead added")


# ------------- PAGE: Leads / Capture -------------
if page == "Leads / Capture":
    st.header("📇 Lead Capture")
    with st.form("lead_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Lead Source", ["Google Ads", "Organic", "Referral", "Phone", "Insurance", "Other"])
            source_details = st.text_input("Source details (UTM / notes)")
            contact_name = st.text_input("Contact name")
            contact_phone = st.text_input("Contact phone")
            contact_email = st.text_input("Contact email")
        with col2:
            property_address = st.text_input("Property address")
            damage_type = st.selectbox("Damage type", ["water", "fire", "mold", "contents", "reconstruction", "other"])
            assigned_to = st.text_input("Assigned to")
            qualified_choice = st.selectbox("Qualified?", ["Yes", "No"], index=0)
            sla_hours = st.number_input("SLA hours", min_value=1, value=24)
        notes = st.text_area("Notes")
        estimated_value = st.number_input("Job Value Estimate (USD)", min_value=0.0, value=0.0, step=100.0)
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
                estimated_value=float(estimated_value or 0.0),
                qualified=(qualified_choice == "Yes")
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


# ------------- PAGE: Pipeline Board (REDESIGN) -------------
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Dashboard — Redesigned")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        df = leads_df(s)
        weights = st.session_state.weights

        # KPIs (2 rows x 4 columns)
        total_leads = int(len(df))
        qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
        total_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
        awarded_leads = int(df[df["status"] == LeadStatus.AWARDED].shape[0]) if not df.empty else 0
        lost_leads = int(df[df["status"] == LeadStatus.LOST].shape[0]) if not df.empty else 0
        closed_leads = awarded_leads + lost_leads
        conversion_rate = (awarded_leads / closed_leads * 100) if closed_leads > 0 else 0.0
        active_leads = total_leads - closed_leads
        avg_value = (total_value / total_leads) if total_leads else 0.0

        KPI_COLORS = [
            ("#2563eb", "Total Leads", total_leads, f"{qualified_leads} qualified"),
            ("#7c3aed", "Pipeline Value", f"${total_value:,.0f}", "Estimated value"),
            ("#16a34a", "Conversion Rate", f"{conversion_rate:.1f}%", f"{awarded_leads}/{closed_leads} won"),
            ("#f97316", "Active Leads", active_leads, "In progress"),
            ("#ef4444", "Awarded", awarded_leads, "Won jobs"),
            ("#e11d48", "Lost", lost_leads, "Lost jobs"),
            ("#06b6d4", "Avg SLA (hrs)", f"{df['sla_hours'].mean():.1f}" if not df.empty else "—", "Avg SLA"),
            ("#0f172a", "Avg Job Value", f"${avg_value:,.0f}", "Average estimate"),
        ]

        # Render 2 rows x 4 columns
        st.markdown("<div style='display:flex;flex-wrap:wrap;'>", unsafe_allow_html=True)
        for idx, (col_color, label, value, note) in enumerate(KPI_COLORS):
            # each card ~24% width -> 4 columns
            st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(90deg, {col_color}, {col_color}); width:24%; margin-right:1%; color: #ffffff;">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-number">{value}</div>
                    <div style="margin-top:8px; color: rgba(255,255,255,0.9); font-size:12px;">{note}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

        # Stage breakdown (2 rows x 4)
        st.markdown("### 📈 Pipeline Stages")
        stage_colors = {
            LeadStatus.NEW: "#2563eb",
            LeadStatus.CONTACTED: "#eab308",
            LeadStatus.INSPECTION_SCHEDULED: "#f97316",
            LeadStatus.INSPECTION_COMPLETED: "#14b8a6",
            LeadStatus.ESTIMATE_SUBMITTED: "#a855f7",
            LeadStatus.AWARDED: "#16a34a",
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
                            <div style="font-weight:700; font-size:14px; color: #fff;">{status}</div>
                            <div style="font-weight:800; font-size:24px; color:{color};">{cnt}</div>
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
            # SLA calculation
            sla_entered = row.get("sla_entered_at") or row.get("created_at")
            if isinstance(sla_entered, str):
                try:
                    sla_entered = datetime.fromisoformat(sla_entered)
                except:
                    sla_entered = datetime.utcnow()
            if pd.isna(sla_entered):
                sla_entered = datetime.utcnow()
            deadline = sla_entered + timedelta(hours=int(row.get("sla_hours") or 24))
            remaining = deadline - datetime.utcnow()
            overdue = remaining.total_seconds() <= 0

            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "No name",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(remaining.total_seconds() / 3600.0),
                "priority_score": score,
                "status": row.get("status"),
                "sla_overdue": overdue,
                "sla_deadline": deadline,
                "damage_type": row.get("damage_type", "Unknown")
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        # Render priority cards (two columns)
        if not pr_df.empty:
            pr_rows = pr_df.head(8).to_dict(orient="records")
            cols = st.columns(2)
            # we'll render one card per column alternately for two-col layout
            for idx, r in enumerate(pr_rows):
                status_color = stage_colors.get(r["status"], "#ffffff")
                if r["priority_score"] >= 0.7:
                    priority_color = "#ef4444"; priority_label = "🔴 CRITICAL"
                elif r["priority_score"] >= 0.45:
                    priority_color = "#f97316"; priority_label = "🟠 HIGH"
                else:
                    priority_color = "#22c55e"; priority_label = "🟢 NORMAL"

                # SLA HTML (time left red)
                if r["sla_overdue"]:
                    sla_html = f"<span style='color:var(--primary-red); font-weight:700;'>❗ OVERDUE</span>"
                else:
                    hours_left = int(r['time_left_hours'])
                    mins_left = int((r['time_left_hours'] * 60) % 60)
                    # time-left must be red per your instruction
                    sla_html = f"<span style='color:var(--primary-red); font-weight:700;'>⏳ {hours_left}h {mins_left}m left</span>"

                card_html = f"""
                <div style="background: #0f172a; color:#fff; border-radius:12px; padding:12px; margin-bottom:10px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;">
                      <div style="margin-bottom:8px;">
                        <span style="color:{priority_color}; font-weight:800;">{priority_label}</span>
                        <span style="margin-left:8px; padding:6px 12px; border-radius:20px; background:{status_color}22; color:{status_color}; font-weight:700;">{r['status']}</span>
                      </div>
                      <div style="font-size:16px; font-weight:800; color:#fff;">#{int(r['id'])} — {r['contact_name']}</div>
                      <div style="color:var(--muted); margin-top:6px;">{r['damage_type'].title()} | Est: <span style='color:var(--money-green); font-weight:800;'>${r['estimated_value']:,.0f}</span></div>
                      <div style="margin-top:8px;">{sla_html}</div>
                    </div>
                    <div style="text-align:right; padding-left:12px;">
                      <div style="font-size:28px; font-weight:900; color:{priority_color};">{r['priority_score']:.2f}</div>
                      <div style="font-size:11px; color:var(--muted); text-transform:uppercase;">Priority</div>
                    </div>
                  </div>
                </div>
                """
                target_col = cols[idx % 2]
                target_col.markdown(card_html, unsafe_allow_html=True)
        else:
            st.info("No priority leads to display.")

        st.markdown("---")

        # All Leads (expandable edit)
        st.markdown("### 📋 All Leads (expand a card to edit / change status — Awarded -> invoice)")
        for lead in leads:
            est_val_disp = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'Unknown'} — {est_val_disp}"
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
                        sla_status_html = "<div style='color:var(--primary-red);font-weight:700;'>❗ OVERDUE</div>"
                    else:
                        hours = int(remaining.total_seconds() // 3600)
                        mins = int((remaining.total_seconds() % 3600) // 60)
                        sla_status_html = f"<div style='color:var(--primary-red);font-weight:700;'>⏳ {hours}h {mins}m</div>"
                    # Lead status label with colors
                    status_color = stage_colors.get(lead.status, "#111827")
                    st.markdown(f"""
                        <div style='text-align:right;'>
                            <div style='display:inline-block; padding:8px 14px; border-radius:20px; background:{status_color}22; color:{status_color}; font-weight:800;'>{lead.status}</div>
                            <div style='margin-top:12px;'>{sla_status_html}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                # Quick contact row
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    with qc1:
                        st.markdown(f"<a href='tel:{phone}'><button class='btn-animated' style='background:var(--primary-blue); color:#fff;'>📞 Call</button></a>", unsafe_allow_html=True)
                    with qc2:
                        wa_number = phone.lstrip("+").replace(" ", "").replace("-", "")
                        wa_link = f"https://wa.me/{wa_number}?text=Hi%2C%20following%20up%20on%20your%20restoration%20request."
                        st.markdown(f"<a href='{wa_link}' target='_blank'><button class='btn-animated' style='background:var(--money-green); color:#fff;'>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(""); qc2.write("")
                if email:
                    with qc3:
                        st.markdown(f"<a href='mailto:{email}?subject=Follow%20up'><button class='btn-animated' style='background:#fff; color:var(--text); border:1px solid #e5e7eb;'>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write("")
                qc4.write("")

                st.markdown("---")
                # Update form
                with st.form(f"update_lead_{lead.id}"):
                    st.markdown("#### Update Lead")
                    u1, u2 = st.columns(2)
                    with u1:
                        new_status = st.selectbox("Status", LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status) if lead.status in LeadStatus.ALL else 0, key=f"status_{lead.id}")
                        new_assigned = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        new_contacted = st.checkbox("Contacted", value=bool(lead.contacted), key=f"contacted_{lead.id}")
                    with u2:
                        insp_sched = st.checkbox("Inspection Scheduled", value=bool(lead.inspection_scheduled), key=f"insp_sched_{lead.id}")
                        insp_comp = st.checkbox("Inspection Completed", value=bool(lead.inspection_completed), key=f"insp_comp_{lead.id}")
                        est_sub = st.checkbox("Estimate Submitted", value=bool(lead.estimate_submitted), key=f"est_sub_{lead.id}")

                    new_notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                    new_est_val = st.number_input("Job Value Estimate (USD)", value=float(lead.estimated_value or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead.id}")

                    awarded_file = None
                    award_comment = ""
                    lost_comment = ""
                    if new_status == LeadStatus.AWARDED:
                        st.markdown("**Award details**")
                        award_comment = st.text_area("Award comment", key=f"award_comment_{lead.id}")
                        awarded_file = st.file_uploader("Upload Invoice File (optional)", type=["pdf","jpg","jpeg","png","xlsx","csv"], key=f"award_inv_{lead.id}")
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
                                if db_lead.sla_entered_at is None:
                                    db_lead.sla_entered_at = datetime.utcnow()
                                if new_status == LeadStatus.AWARDED:
                                    db_lead.awarded_date = datetime.utcnow()
                                    db_lead.awarded_comment = award_comment
                                    if awarded_file is not None:
                                        path = save_uploaded_file(awarded_file, prefix=f"lead_{db_lead.id}_inv")
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
                            st.write(traceback.format_exc())


# ------------- PAGE: Analytics & SLA (Donut using CSS) -------------
elif page == "Analytics & SLA":
    st.header("📈 Funnel Analytics & SLA")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No leads to analyze.")
    else:
        # status counts
        try:
            status_counts = df["status"].value_counts().reindex(LeadStatus.ALL, fill_value=0)
        except Exception:
            status_counts = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0)

        # Render donut via CSS conic-gradient (no plotly)
        labels = status_counts.index.tolist()
        vals = status_counts.values.tolist()
        total = sum(vals) or 1
        # build slices in deg
        degrees = [v / total * 360 for v in vals]
        # build CSS gradient string
        start = 0
        parts = []
        color_map = [stage_colors[s] for s in labels]
        for deg, col in zip(degrees, color_map):
            end = start + deg
            parts.append(f"{col} {start}deg {end}deg")
            start = end
        gradient_css = ", ".join(parts)
        donut_html = f"""
        <div style='display:flex; align-items:center; gap:20px;'>
          <div style='width:260px; height:260px; border-radius:50%; background: conic-gradient({gradient_css}); position:relative;'>
            <div style='position:absolute; left:20%; top:20%; width:60%; height:60%; border-radius:50%; background:#fff; display:flex; align-items:center; justify-content:center;'>
              <div style='text-align:center; color:#0b1220; font-weight:700;'>{total} leads</div>
            </div>
          </div>
          <div style='flex:1;'>
            <div style='font-weight:700; margin-bottom:10px;'>Stage breakdown</div>
        """
        for lbl, v, col in zip(labels, vals, color_map):
            pct = (v / total * 100) if total else 0
            donut_html += f"""
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>
              <div style='display:flex; align-items:center; gap:8px;'>
                <div style='width:14px; height:14px; border-radius:4px; background:{col};'></div>
                <div style='font-weight:600'>{lbl}</div>
              </div>
              <div style='color:var(--muted);'>{v} ({pct:.0f}%)</div>
            </div>
            """
        donut_html += "</div></div>"
        st.markdown(donut_html, unsafe_allow_html=True)

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


# ------------- PAGE: Exports -------------
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
        "id": e.id, "lead_id": e.lead_id, "amount": e.amount, "created_at": e.created_at, "approved": e.approved, "lost": e.lost
    } for e in s.query(Estimate).all()])
    if not df_est.empty:
        st.download_button("Download estimates.csv", df_est.to_csv(index=False).encode("utf-8"), file_name="estimates.csv", mime="text/csv")

# EOF
