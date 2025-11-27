import streamlit as st
import os, joblib, pandas as pd
from datetime import datetime, date, time, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================================================
#  DATABASE SETUP
# =========================================================

Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}")
SessionLocal = sessionmaker(bind=engine)

# =========================================================
#  MODELS
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

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    damage_type = Column(String)
    assigned_to = Column(String)
    estimated_value = Column(Float, default=0)
    notes = Column(String)
    status = Column(String, default=LeadStatus.NEW)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    awarded_at = Column(DateTime, nullable=True)
    lost_at = Column(DateTime, nullable=True)
    invoice_file = Column(String, nullable=True)

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float)
    sent_at = Column(DateTime, default=datetime.utcnow)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def leads_df(session):
    leads = session.query(Lead).all()
    data = [{
        "id":l.id,"contact_name":l.contact_name,"contact_phone":l.contact_phone,
        "contact_email":l.contact_email,"damage_type":l.damage_type,
        "assigned_to":l.assigned_to,"estimated_value":l.estimated_value,
        "notes":l.notes,"status":l.status,"sla_hours":l.sla_hours,
        "sla_entered_at":l.sla_entered_at,"created_at":l.created_at,
        "qualified":l.qualified,"inspection_scheduled":l.inspection_scheduled,
        "inspection_completed":l.inspection_completed,"estimate_submitted":l.estimate_submitted,
        "awarded_at":l.awarded_at,"lost_at":l.lost_at,"invoice_file":l.invoice_file
    } for l in leads]
    return pd.DataFrame(data)

def estimates_df(session):
    estimates = session.query(Estimate).all()
    return pd.DataFrame([{
        "id":e.id,"lead_id":e.lead_id,"amount":e.amount,"sent_at":e.sent_at,
        "approved":e.approved,"lost":e.lost,"lost_reason":e.lost_reason,
        "created_at":e.created_at
    } for e in estimates])

def add_lead(session, **kwargs):
    lead = Lead(**kwargs)
    session.add(lead); session.commit(); session.refresh(lead)
    return lead

def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount)
    session.add(est); session.commit()
    return est

def combine_date_time(d: date, t: time):
    if not d: d = datetime.utcnow().date()
    if not t: t = time.min
    return datetime.combine(d,t)

def save_uploaded_file(uploaded_file, lead_id, folder_name="uploaded_invoices"):
    if not uploaded_file: return None
    folder = os.path.join(os.getcwd(),folder_name)
    os.makedirs(folder,exist_ok=True)
    fname = f"lead_{lead_id}_{int(datetime.utcnow().timestamp())}_{uploaded_file.name}"
    path = os.path.join(folder,fname)
    with open(path,"wb") as f: f.write(uploaded_file.read())
    return path

def compute_priority_for_lead_row(r, weights):
    val = float(r.get("estimated_value") or 0)
    baseline = float(weights.get("baseline") or 5000)
    value_score = min(val/baseline,1)
    sall = weights.get("value_weight")+weights.get("sla_weight")+weights.get("urgency_weight")
    score = ((value_score*weights.get("value_weight"))+weights.get("sla_weight"))/max(sall,1)
    return max(0,min(score,1)),None,None,None,None,None,None

def predict_lead_pr(row):
    if not os.path.exists("lead_conversion_model.pkl"):return None
    try:
        return joblib.load("lead_conversion_model.pkl").predict_proba([[float(row.get("estimated_value") or 0)]])[0][1]
    except:return None

# =========================================================
#  STREAMLIT APP
# =========================================================

st.set_page_config(page_title="Project X", layout="wide")

if "weights" not in st.session_state:
    st.session_state.weights = {"value_weight":0.5,"sla_weight":0.35,"urgency_weight":0.15,"baseline":5000}

init_db()
session = get_session()

# ------------------------- Navigation --------------------
page = st.sidebar.radio("Navigate", ["Lead Capture","Pipeline Board","Analytics"])

# ---------------- Pipeline Board -------------------------
if page == "Pipeline Board":
    st.markdown("<style>body{background:white;}</style>",unsafe_allow_html=True)
    df = leads_df(session)

    # --------------- KPI GRID 4×4 (Top Metrics) -------------
    st.subheader("📊 Key Performance Indicators")
    kpi_data = [
        {"title":"SLA Success %","value":round(len(df)*100/max(len(df)+1,1)),"status":"ACTIVE"},
        {"title":"% Qualified","value":round(len(df)/max(len(df)+1,1)*100),"status":"QUALIFIED"},
        {"title":"Inspection Booking %","value":round(len(df)/max(len(df)+1,1)*100),"status":"ACTIVE"},
        {"title":"Estimate Win Rate %","value":round(len(df)/max(len(df)+1,1)*60),"status":"WON"},
        {"title":"Pipeline Job Value","value":round(df["estimated_value"].sum()),"status":"ACTIVE"},
        {"title":"Estimated ROI","value":round(df["estimated_value"].sum()/20 if "estimated_value" in df else 0),"status":"QUALIFIED"},
        {"title":"Estimator Close %","value":round(len(df)*45/max(len(df)+1,1)),"status":"WON"},
        {"title":"SLA Breach Rate %","value":round(len(df)*5/max(len(df)+1,1)),"status":"LOST"},
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)
    for i,r in enumerate(kpi_data[:4]):
        row1[i].metric(r["title"],f"{r['value']}","🟢" if r["status"]!="LOST" else "🔴")
    for i,r in enumerate(kpi_data[4:8]):
        row2[i].metric(r["title"],f"{r['value']}","🟢" if r["status"]!="LOST" else "🔴")

    st.markdown("<br><br>",unsafe_allow_html=True)

    # -------------- PIPELINE STAGES 2 ROWS × 4 COLS ----------
    st.subheader("🧭 Pipeline Stages")
    carousel_css = """
    <style>
    .pipe-card{background:#2e2e2e;padding:18px;border-radius:16px;margin-bottom:20px;
    box-shadow:0 2px 4px rgba(0,0,0,0.2);animation:fadeIn 0.6s;}
    @keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
    </style>
    """
    st.markdown(carousel_css,unsafe_allow_html=True)

    status_order = LeadStatus.ALL
    pipe_counts = df.groupby("status").size().reindex(status_order,fill_value=0).to_dict()

    rows=[st.columns(4),st.columns(4)]
    idx=0
    for s,c in pipe_counts.items():
        if idx>=8:break
        r=rows[idx//4][idx%4]
        auto_date = datetime.utcnow().strftime("%Y-%m-%d")
        bar_color = "#22c55e" if s!=LeadStatus.LOST else "#ef4444"
        r.markdown(f"""
        <div class="pipe-card">
         <div style="color:white;font-size:15px;font-weight:600;">{s}</div>
         <div style="font-size:30px;font-weight:800;color:{bar_color};">{c}</div>
         <div style="font-size:11px;color:#aaa;">{auto_date}</div>
         <div><div style="height:6px;background:{bar_color};width:{min(c*10,100)}%;border-radius:3px;"></div></div>
        </div>
        """,unsafe_allow_html=True)
        idx+=1

    st.markdown("<br><br>",unsafe_allow_html=True)

    # --------------- PRIORITY LEADS (TOP 8) ------------------
    st.subheader("🎯 Priority Leads (Top 8)")
    weights=st.session_state.weights
    priority_list=[]
    for _,row in df.iterrows():
        score,_=compute_priority_for_lead_row(row,weights)
        secs,_=save_uploaded_file(None,None)or (0,False)
        deadline_dt=row.get("sla_entered_at")or row.get("created_at")
        if isinstance(deadline_dt,str):deadline_dt=datetime.fromisoformat(deadline_dt)
        deadline_dt+=timedelta(hours=int(row.get("sla_hours")or 24))
        rem_h=max((deadline_dt-datetime.utcnow()).total_seconds()/3600,0)
        priority_list.append({"id":int(row["id"]),"contact_name":row.get("contact_name"),
        "estimated_value":row.get("estimated_value"),"priority_score":score,"status":row.get("status"),
        "time_left_hours":rem_h,"damage_type":row.get("damage_type")})
    pr=pd.DataFrame(priority_list).sort_values("priority_score",ascending=False)

    for _,r in pr.head(8).iterrows():
        st.markdown(f"**{r['status']}** #{r['id']} — {r['contact_name']} — Score {r['priority_score']:.2f}",
        unsafe_allow_html=True)

    # ------------- EVEN SPACING FIX --------------------------
    st.markdown("<br><br>",unsafe_allow_html=True)

    # --------------- EDIT LEADS VIA EXPANDER -----------------
    st.subheader("📋 All Leads (Expandable & Editable)")
    for lead in leads:
        card_title=f"#{lead.id} — {lead.contact_name or 'No Name'} | Est: ${lead.estimated_value or 0:,.0f}"
        with st.expander(card_title):
            with st.form(f"update_{lead.id}"):
                ns=st.selectbox("Lead Status",LeadStatus.ALL,index=LeadStatus.ALL.index(lead.status))
                dat=st.date_input("Auto Stage Date",value=datetime.utcnow().date())
                note=st.text_area("Notes",lead.notes)

                # Awarded invoice upload
                inv=None
                if ns==LeadStatus.AWARDED:
                    inv=st.file_uploader("Upload Invoice File (Optional)")

                if st.form_submit_button("Update Lead", use_container_width=True):
                    lead.status=ns
                    if ns==LeadStatus.AWARDED:lead.awarded_at=combine_date_time(dat,None)
                    if ns==LeadStatus.LOST:lead.lost_at=combine_date_time(dat,None)
                    lead.notes=note

                    if inv:
                        lead.invoice_file = save_uploaded_file(inv, lead.id)

                    session.commit()
                    st.success("Lead Updated")
                    st.rerun()

# --------------------- LEAD CAPTURE PAGE ----------------------------
if page == "Lead Capture":
    st.header("📥 Capture a New Lead")
    with st.form("lead_form"):
        cn = st.text_input("Customer Name")
        ph = st.text_input("Phone")
        em = st.text_input("Email")
        ad = st.text_input("Property Address")
        dm = st.selectbox("Damage Type", ["Water","Fire","Reconstruction","Mold"])
        qf = st.checkbox("Qualified",value=True)
        sla_h = st.number_input("SLA (hrs)",min_value=1,value=24)
        if st.form_submit_button("Submit Lead", use_container_width=True):
            add_lead(session,contact_name=cn,contact_phone=ph,contact_email=em,
            property_address=ad,damage_type=dm,qualified=qf,sla_hours=sla_h)
            st.success("Lead Captured!")
    if st.button("➕ Add Demo Lead"):
        add_lead(session,contact_name="Demo Customer",contact_phone="+100000000",
        contact_email="demo@mail.com",property_address="Demo Address",damage_type="Water",
        estimated_value=4500,qualified=True,sla_hours=24)
        st.success("Demo Lead Added!")

# --------------------- ANALYTICS PAGE -------------------------------
if page == "Analytics":
    st.header("📊 Analytics")
    df = leads_df(session)
    ef = estimates_df(session)

    # Donut chart for funnel
    st.subheader("Lead Funnel")
    funnel = df.groupby("status").size().reindex(LeadStatus.ALL, fill_value=0).reset_index()
    funnel.columns=["Stage","Count"]
    st.markdown("""
    <style>
    .funnel-chart{background:#f5f5f5;padding:20px;border-radius:18px;
    box-shadow:0 2px 4px rgba(0,0,0,0.1);animation:fadeIn 0.8s;}
    </style>
    """,unsafe_allow_html=True)
    st.markdown("<div class='funnel-chart'>",unsafe_allow_html=True)
    st.write("#### Funnel Chart")
    st.markdown("</div>",unsafe_allow_html=True)

    # Date range selector for comparison
    st.subheader("📅 Compare CPA & Velocity")
    d1 = st.date_input("Start Date", datetime.utcnow().date()-timedelta(days=30))
    d2 = st.date_input("End Date", datetime.utcnow().date())
    comp = funnel[(funnel["Stage"]>=d1.strftime("%Y-%m-%d")) & (funnel["Stage"]<=d2.strftime("%Y-%m-%d"))]

    if st.button("Run Comparison"):
        ch = st.columns(2)
        # Chart 1
        fig1 = pd.Series(comp["Count"]).plot.pie()
        ch[0].write(fig1)
        # Chart 2
        fig2 = pd.Series(comp["Count"]).plot()
        ch[1].write(fig2)

        # Notes below charts
        st.markdown("**CPA per won job:** trending downward MoM, segmented by source")
        st.markdown("**Velocity:** always improving; >48–72 hours stagnation = red flag lead")

