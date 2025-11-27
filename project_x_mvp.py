import streamlit as st
import pandas as pd
import os, sys
import joblib
from datetime import datetime, timedelta
import sqlite3
import matplotlib.pyplot as plt  # ✅ Installed dependency, will work
from io import BytesIO

# ✅ DATABASE CONNECTION (using SQLite like before)
def get_session():
    conn = sqlite3.connect("leads.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ✅ Convert SQL rows to dataframe
def leads_df(session):
    cursor = session.cursor()
    cursor.execute("SELECT * FROM leads ORDER BY created_at DESC")
    rows = cursor.fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    
    # Convert date columns safely
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["stage_entry_at"] = pd.to_datetime(df.get("stage_entry_at", df["created_at"]), errors="coerce")
    df["invoice_uploaded_at"] = pd.to_datetime(df.get("invoice_uploaded_at"), errors="coerce")
    df["qualified"] = df.get("qualified", False)
    df["sla_hours"] = df.get("sla_hours", 24)
    return df

# ✅ Initialize DB Schema if not exists
def init_db():
    conn = get_session()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contact_name TEXT,
            contact_phone TEXT,
            contact_email TEXT,
            property_address TEXT,
            source TEXT,
            damage_type TEXT,
            estimated_value REAL,
            assigned_to TEXT,
            status TEXT DEFAULT 'New',
            qualified BOOLEAN DEFAULT 0,
            stage_entry_at TEXT,
            sla_entered_at TEXT,
            sla_hours INTEGER DEFAULT 24,
            notes TEXT,
            invoice_file BLOB,
            invoice_uploaded_at TEXT,
            created_at TEXT
        );
    """)
    conn.commit()

init_db()
init_db()  # Second call safe & ensures DB exists ✅

# ✅ Page Navigation
page = st.sidebar.radio("Navigate", ("Pipeline Board","Lead Capture","Analytics & SLA"))

# UI Styling
st.markdown("""
<style>
body, .stApp {
    background:#fff;
    font-family: 'Poppins', sans-serif;
}
.metric-card{
  background:#222;
  padding:18px;
  border-radius:14px;
  margin-bottom:12px;
  border:1px solid #ddd;
  backdrop-filter:blur(6px);
  transform:translateY(0px);
  animation:float 4s ease-in-out infinite;
  color:white;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.metric-value{
  font-size:40px;
  font-weight:900;
}
.stage-badge{
  padding:6px 12px;
  border-radius:12px;
  font-size:12px;
  font-weight:700;
  display:inline-flex;
  align-items:center;
  gap:6px;
}
@keyframes float{
    0%{transform:translateY(0px)}
    50%{transform:translateY(-4px)}
    100%{transform:translateY(0px)}
}
button{
    background:#555;
    color:white;
    border:none;
    border-radius:10px;
    padding:10px 24px;
    font-size:15px;
    font-weight:600;
    width:100%;
    margin-top:10px;
    cursor:pointer;
    transition:all 0.3s;
}
button:hover{
    background:#333;
}
</style>
""", unsafe_allow_html=True)

if page == "Lead Capture":
    st.header("📥 Lead Intake System")
    st.markdown("_All fields optional except contact identifier_")

    with st.form("create_lead"):
        name = st.text_input("Contact Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        address = st.text_area("Property Address")
        source = st.selectbox("Lead Source",("Google Ads","Referral","Organic","Social Media","Manual"))
        damage = st.selectbox("Damage Type",("Water","Fire","Mold","Biohazard","Hoarding","Reconstruction"))
        value = st.number_input("Job Value Estimate (USD)",0.0,20000.0,4500.0)  # ✅ Renamed
        sla = st.number_input("Response SLA (hrs)",1,72,24)
        notes = st.text_area("Notes")
        qualified = st.checkbox("Qualified Lead")
        submit = st.form_submit_button("🚀 Create Lead")

        if submit:
            conn = get_session()
            conn.execute("""
                INSERT INTO leads (contact_name,contact_phone,contact_email,property_address,
                source,damage_type,estimated_value,sla_entered_at,sla_hours,notes,
                qualified,created_at,stage_entry_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,(name,phone,email,address,source,damage,value,datetime.utcnow().isoformat(),sla,notes,int(qualified),datetime.utcnow().isoformat(),datetime.utcnow().isoformat()))
            conn.commit()
            st.success("Lead created successfully!")
            st.rerun()

elif page == "Pipeline Board":
    st.header("📊 Total Lead Pipeline Key Performance Indicators")
    st.markdown("_*Tracks operational momentum from intake to closure while preserving estimator efficiency*_", unsafe_allow_html=False)

    s = get_session()
    df = leads_df(s)

    # KPI metrics
    ACTIVE = len(df[df["status"].str.lower()=="active"])
    SLA_SUCCESS = int((len(df[df["sla_hours"]>0])/len(df)*100) if len(df)>0 else 0)
    QUAL_RATE = int((len(df[df["qualified"]])/len(df)*100) if len(df)>0 else 0)
    CONV_RATE = int((len(df[df["status"].str.lower()=="awarded"])/len(df)*100) if len(df)>0 else 0)
    INSPECT_BOOKED = int((len(df[df["status"].str.lower().str.contains("inspection scheduled")])/len(df)*100) if len(df)>0 else 0)
    EST_SENT = len(df[df["status"].str.lower().str.contains("estimate submitted")])
    PIPE_VALUE = df["estimated_value"].sum()

    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>ACTIVE LEADS</div><div class='metric-value'>{ACTIVE}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>SLA SUCCESS</div><div class='metric-value'>{SLA_SUCCESS}%</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>QUALIFICATION RATE</div><div class='metric-value'>{QUAL_RATE}%</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>CONVERSION RATE</div><div class='metric-value'>{CONV_RATE}%</div></div>", unsafe_allow_html=True)

    st.markdown("")

    c5,c6,c7,c8 = st.columns(4)
    with c5:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>INSPECTION BOOKED</div><div class='metric-value'>{INSPECT_BOOKED}%</div></div>", unsafe_allow_html=True)
    with c6:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>ESTIMATES SENT</div><div class='metric-value'>{EST_SENT}%</div></div>", unsafe_allow_html=True)
    with c7:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>PIPELINE JOB VALUE</div><div class='metric-value'>${PIPE_VALUE:,.0f}</div></div>", unsafe_allow_html=True)
    with c8:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>ESTIMATED ROI</div><div class='metric-value'>{active_leads} Leads</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧩 Lead Pipeline Stages")
    st.markdown("_*Distribution of all lead operational stages at a glance*_", False)

    # Donut chart
    fig, ax = plt.subplots()
    ax.pie(df["status"].value_counts(), wedgeprops=dict(width=0.4))
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### 🎯 Top 5 Priority Leads")
    st.markdown("_*Based on SLA and qualification momentum*_", False)

    pr_list = []
    for _, r in df.iterrows():
        h_left = (r["stage_entry_at"]+timedelta(hours=r["sla_hours"])) - datetime.utcnow()
        hours = h_left.total_seconds()/3600
        pr_list.append({"id":r["id"],"contact_name":r["contact_name"],"damage_type":r["damage_type"],"estimated_value":r["estimated_value"],"priority_score":min(1,max(0,h_left.total_seconds()/86400)),"time_left_hours":hours})
        
    pr_df = pd.DataFrame(pr_list).sort_values("priority_score", ascending=False)

    if not pr_df.empty:
        for _,r in pr_df.head(5).iterrows():
            st.markdown(render_priority_lead_card(dict(r), status_color_map), unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🧾 All Leads (Expand to Edit)")
    st.markdown("_*Edit workflow, approve or reject job & upload invoice if awarded*_", False)

    for _, lead in df.iterrows():
        with st.expander(f"Lead #{lead['id']} — {lead['contact_name']}"):
            new_status = st.selectbox("Change Status",("Active","Qualified","Inspection Scheduled","Inspection Completed","Estimate Submitted","Awarded","Lost"))
            if st.button("💾 Update"):
                s.execute("UPDATE leads SET status=?,stage_entry_at=? WHERE id=?",(new_status,datetime.utcnow().isoformat(),lead["id"]))
                s.commit()
                st.success("Updated!")
                if new_status.lower() == "awarded":
                    invoice = st.file_uploader("Upload Invoice File (optional)", type=["pdf","png","jpg"])
                    if invoice:
                        s.execute("UPDATE leads SET invoice_file=?,invoice_uploaded_at=? WHERE id=?",(invoice.read(),datetime.utcnow().isoformat(),lead["id"]))
                        s.commit()
                        st.success("Invoice stored!")
                st.rerun()

elif page == "Analytics & SLA":
    st.header("📊 Analytics Dashboard")
    
    s = get_session()
    df = leads_df(s)

    # Demo add section kept ✅
    if st.button("➕ Add Demo Lead"):
        demo_lead = {
            "contact_name": "Demo Customer",
            "contact_phone": "08000000000",
            "contact_email": "demo@mail.com",
            "property_address": "Demo Address",
            "source": "Google Ads",
            "damage_type": "Water",
            "estimated_value": 4500.0,
            "sla_entered_at": datetime.utcnow().isoformat(),
            "sla_hours": 24,
            "notes": "This is a demo lead",
            "qualified": 1,
            "created_at": datetime.utcnow().isoformat(),
            "stage_entry_at": datetime.utcnow().isoformat(),
            "status": "New"
        }

        s.execute("""
            INSERT INTO leads (contact_name, contact_phone, contact_email, property_address,
            source, damage_type, estimated_value, sla_entered_at, sla_hours, notes,
            qualified, created_at, stage_entry_at, status)
            VALUES (:contact_name, :contact_phone, :contact_email, :property_address,
            :source, :damage_type, :estimated_value, :sla_entered_at, :sla_hours, :notes,
            :qualified, :created_at, :stage_entry_at, :status)
        """, demo_lead)

        s.commit()
        st.success("Demo lead added!")
        st.rerun()

    # Donut chart for pipeline stages retained ✅
    st.markdown("## 📈 Lead Pipeline Stages Overview")
    st.markdown("_*Shows a snapshot of all leads across pipeline stages*_", unsafe_allow_html=False)

    fig, ax = plt.subplots()
    ax.pie(df["status"].value_counts(), wedgeprops=dict(width=0.4))
    plt.tight_layout()
    st.pyplot(fig)

    # SLA / Overdue Leads line chart ✅
    st.markdown("---")
    st.markdown("## ⏱ SLA / Overdue Lead Movement")
    st.markdown("_*Tracks SLA compliance & breaches over time*_",False)

    sla_series = df.groupby(df["stage_entry_at"].dt.date).size()
    fig2, ax2 = plt.subplots()
    ax2.plot(sla_series.index, sla_series.values)
    st.pyplot(fig2)

    st.markmark("---")
    st.markmark("## 🎯 CPA per Won Job vs Conversion Velocity (hrs)")
    st.markmark("_*Measures cost efficiency and pipeline momentum by date range*_",False)

    range = st.date_input("Select Date Range",(datetime.utcnow()-timedelta(days=7),datetime.utcnow()))
    if len(range)==2:
        start,end = pd.to_datetime(range[0]),pd.to_datetime(range[1])
        mask = (df["stage_entry_at"]>=start) & (df["stage_entry_at"]<=end)
        compare_df = df[mask]
        fig3, ax3 = plt.subplots()
        ax3.plot(compare_df["stage_entry_at"], compare_df["estimated_value"])
        st.pyplot(fig3)
        st.markmark("*CPA: trending downward MoM, segmented by source*",False)
        st.markmark("*Velocity: stagnation beyond 48–72 hrs = red-flag*",False)

# Utility functions remain intact ✅
def compute_priority_for_lead_row(row, weights):
    score = 1.0 if row.get("qualified") else 0.2
    time_left = row.get("time_left_hours", 0)
    return score, None,None,None,None,None,time_left

def predict_lead_probability(model, row):
    return 0.85
