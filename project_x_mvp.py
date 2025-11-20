import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# --- DB setup
Base = declarative_base()
engine = create_engine('sqlite:///leads.db')
Session = sessionmaker(bind=engine)

# --- Models
class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    source = Column(String)
    notes = Column(Text)
    assigned_to = Column(String)
    estimated_value = Column(Float)
    status = Column(String, default="New")
    sla_stage = Column(String)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime)
    inspection_completed = Column(Boolean, default=False)
    inspection_completed_at = Column(DateTime)
    estimate_submitted = Column(Boolean, default=False)
    awarded_comment = Column(Text)
    awarded_date = Column(DateTime)
    lost_comment = Column(Text)
    lost_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer, ForeignKey("leads.id"))
    amount = Column(Float)
    details = Column(Text)
    sent_at = Column(DateTime)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(Text)
    lead = relationship("Lead", backref="estimates")

Base.metadata.create_all(engine)

# --- Helpers
class LeadStatus:
    ALL = ["New","Contacted","Inspection","Estimate","Awarded","Lost"]

def get_session():
    return Session()

def leads_df(session):
    leads = session.query(Lead).all()
    data = []
    for l in leads:
        data.append({
            "id": l.id, "contact_name": l.contact_name, "contact_phone": l.contact_phone,
            "contact_email": l.contact_email, "property_address": l.property_address,
            "damage_type": l.damage_type, "source": l.source, "notes": l.notes,
            "assigned_to": l.assigned_to, "estimated_value": l.estimated_value,
            "status": l.status, "sla_hours": l.sla_hours, "sla_entered_at": l.sla_entered_at,
            "contacted": l.contacted, "inspection_scheduled": l.inspection_scheduled,
            "inspection_scheduled_at": l.inspection_scheduled_at,
            "inspection_completed": l.inspection_completed,
            "inspection_completed_at": l.inspection_completed_at,
            "estimate_submitted": l.estimate_submitted,
            "awarded_comment": l.awarded_comment, "awarded_date": l.awarded_date,
            "lost_comment": l.lost_comment, "lost_date": l.lost_date,
            "created_at": l.created_at
        })
    return pd.DataFrame(data)

def compute_priority_for_lead_row(row, weights):
    value_score = float(row.get("estimated_value") or 0)/1000.0
    sla_hours = row.get("sla_hours") or 24
    sla_entered = row.get("sla_entered_at") or datetime.utcnow()
    if isinstance(sla_entered,str):
        sla_entered = datetime.fromisoformat(sla_entered)
    elapsed = (datetime.utcnow() - sla_entered).total_seconds()/3600
    time_left_h = max(sla_hours - elapsed, 0)
    sla_score = (time_left_h / sla_hours) if sla_hours else 0
    contacted_flag = 1.0 if row.get("contacted") else 0.0
    inspection_flag = 1.0 if row.get("inspection_completed") else 0.0
    estimation_flag = 1.0 if row.get("estimate_submitted") else 0.0
    score = (weights.get("value",0.4)*value_score + weights.get("sla",0.3)*sla_score +
             weights.get("contacted",0.1)*contacted_flag + weights.get("inspection",0.1)*inspection_flag +
             weights.get("estimate",0.1)*estimation_flag)
    return score, value_score, sla_score, contacted_flag, inspection_flag, estimation_flag, time_left_h

# --- Estimate helpers
def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount, details=details, sent_at=None)
    session.add(est)
    session.commit()

def mark_estimate_sent(session, est_id):
    est = session.query(Estimate).get(est_id)
    if est:
        est.sent_at = datetime.utcnow()
        session.commit()

def mark_estimate_approved(session, est_id):
    est = session.query(Estimate).get(est_id)
    if est:
        est.approved = True
        est.lead.status = "Awarded"
        session.commit()

def mark_estimate_lost(session, est_id, reason=None):
    est = session.query(Estimate).get(est_id)
    if est:
        est.lost = True
        est.lost_reason = reason
        est.lead.status = "Lost"
        session.commit()

# --- Streamlit app
st.set_page_config(page_title="Project X CRM", layout="wide")
if "weights" not in st.session_state:
    st.session_state.weights = {"value":0.4,"sla":0.3,"contacted":0.1,"inspection":0.1,"estimate":0.1}

pages = ["Lead Capture","Pipeline Board","Analytics","Export Data"]
page = st.sidebar.radio("Navigation", pages)

# --- Lead Capture
if page == "Lead Capture":
    st.header("📝 Lead Capture")
    s = get_session()
    with st.form("new_lead"):
        contact_name = st.text_input("Contact Name")
        phone = st.text_input("Contact Phone")
        email = st.text_input("Contact Email")
        address = st.text_input("Property Address")
        damage = st.text_input("Damage Type")
        source = st.text_input("Source")
        notes = st.text_area("Notes")
        assigned_to = st.text_input("Assigned To")
        est_val = st.number_input("Estimated Value", min_value=0.0, value=0.0, step=50.0)
        submit = st.form_submit_button("Add Lead")
        if submit:
            lead = Lead(contact_name=contact_name.strip() or None,
                        contact_phone=phone.strip() or None,
                        contact_email=email.strip() or None,
                        property_address=address.strip() or None,
                        damage_type=damage.strip() or None,
                        source=source.strip() or None,
                        notes=notes.strip() or None,
                        assigned_to=assigned_to.strip() or None,
                        estimated_value=est_val or None)
            s.add(lead)
            s.commit()
            st.success(f"Lead #{lead.id} added")
            st.experimental_rerun()

# --- Pipeline Board (FULLY editable + priority)
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()
    if not leads:
        st.info("No leads yet.")
    else:
        # Compute priorities
        df = leads_df(s)
        weights = st.session_state.weights
        priority_list = []
        for _, row in df.iterrows():
            score, value_score, sla_score, contacted_flag, inspection_flag, estimation_flag, time_left_h = compute_priority_for_lead_row(row, weights)
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left_h),
                "priority_score": score,
                "status": row.get("status"),
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        # Priority summary
        st.subheader("Priority Leads")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                color = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                html_block = f"""
                <div style='padding:8px;border-radius:8px;margin-bottom:6px;border:1px solid rgba(255,255,255,0.04);'>
                    <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                    <span style='color:var(--muted);'> | Est: ${r['estimated_value']:,.0f}</span>
                    <span style='color:var(--muted);'> | Time left: {int(r['time_left_hours'])}h</span>
                    <span style='float:right;color:{color};'>Priority: {r['priority_score']:.2f}</span>
                </div>
                """
                st.markdown(html_block, unsafe_allow_html=True)
        else:
            st.info("No priority leads yet.")

        st.markdown("---")

        # Pipeline columns
        cols = st.columns(len(LeadStatus.ALL))
        buckets = {stg: [] for stg in LeadStatus.ALL}
        for l in leads:
            buckets[l.status].append(l)

        for i, stg in enumerate(LeadStatus.ALL):
            with cols[i]:
                st.markdown(f"### {stg} ({len(buckets[stg])})")
                for lead in buckets[stg]:
                    with st.expander(f"#{lead.id} · {lead.contact_name or 'No name'} · ${lead.estimated_value or 0:,.0f}"):
                        # Lead info
                        st.write(f"**Source:** {lead.source} · **Assigned:** {lead.assigned_to}")
                        st.write(f"**Address:** {lead.property_address}")
                        st.write(f"**Damage:** {lead.damage_type}")
                        st.write(f"**Notes:** {lead.notes}")
                        st.write(f"**Created:** {lead.created_at}")

                        # Quick contact
                        qc_cols = st.columns(3)
                        phone = (lead.contact_phone or "").strip()
                        email = (lead.contact_email or "").strip()
                        if phone:
                            tel_link = f"tel:{phone}"
                            wa_link = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text=Hi"
                            qc_cols[0].markdown(f"<a href='{tel_link}'><button>📞 Call</button></a>", unsafe_allow_html=True)
                            qc_cols[1].markdown(f"<a href='{wa_link}' target='_blank'><button>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                        if email:
                            mail_link = f"mailto:{email}?subject=Follow up"
                            qc_cols[2].markdown(f"<a href='{mail_link}'><button>✉️ Email</button></a>", unsafe_allow_html=True)

                        # SLA countdown
                        entered = lead.sla_entered_at or lead.created_at
                        if isinstance(entered,str):
                            entered = datetime.fromisoformat(entered)
                        deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                        remaining = deadline - datetime.utcnow()
                        if remaining.total_seconds() <= 0:
                            st.markdown(f"❗ **SLA OVERDUE** (was due {deadline.strftime('%Y-%m-%d %H:%M')})")
                        else:
                            st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                        st.markdown("---")
                        # Editable form
                        with st.form(f"lead_edit_{lead.id}"):
                            contact_name = st.text_input("Contact name", value=lead.contact_name or "", key=f"cname_{lead.id}")
                            contact_phone = st.text_input("Contact phone", value=lead.contact_phone or "", key=f"cphone_{lead.id}")
                            contact_email = st.text_input("Contact email", value=lead.contact_email or "", key=f"cemail_{lead.id}")
                            assigned_to = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                            est_val = st.number_input("Estimated value", min_value=0.0, value=float(lead.estimated_value or 0.0), step=50.0, key=f"est_{lead.id}")
                            notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")
                            contacted_choice = st.selectbox("Contacted?", ["No","Yes"], index=1 if lead.contacted else 0, key=f"cont_{lead.id}")
                            new_stage = st.selectbox("Move lead stage", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"stage_{lead.id}")
                            save = st.form_submit_button("Save Lead")
                            if save:
                                lead.contact_name = contact_name.strip() or None
                                lead.contact_phone = contact_phone.strip() or None
                                lead.contact_email = contact_email.strip() or None
                                lead.assigned_to = assigned_to.strip() or None
                                lead.estimated_value = float(est_val) if est_val else None
                                lead.notes = notes.strip() or None
                                lead.contacted = True if contacted_choice=="Yes" else False
                                if new_stage != lead.status:
                                    lead.status = new_stage
                                    lead.sla_stage = new_stage
                                    lead.sla_entered_at = datetime.utcnow()
                                s.add(lead)
                                s.commit()
                                st.success(f"Lead #{lead.id} updated")
                                st.rerun()

# --- Analytics
elif page == "Analytics":
    st.header("📊 Analytics")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No data yet.")
    else:
        st.subheader("Leads by Status")
        st.bar_chart(df["status"].value_counts())
        st.subheader("Estimated Value by Status")
        st.bar_chart(df.groupby("status")["estimated_value"].sum())

# --- Export Data
elif page == "Export Data":
    st.header("📤 Export Data")
    s = get_session()
    df = leads_df(s)
    if df.empty:
        st.info("No data yet.")
    else:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "leads.csv", "text/csv")
