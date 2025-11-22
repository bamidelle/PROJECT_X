import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Project X CRM", layout="wide")

# -----------------------------
# Session State Initialization
# -----------------------------
if "leads" not in st.session_state:
    st.session_state.leads = pd.DataFrame(columns=[
        "id", "name", "service", "amount", "stage", "priority", "created"
    ])

if "id_counter" not in st.session_state:
    st.session_state.id_counter = 1


# -----------------------------
# Helper Functions
# -----------------------------
def add_lead(name, service, amount, priority):
    new_id = st.session_state.id_counter
    st.session_state.id_counter += 1

    new_lead = {
        "id": new_id,
        "name": name,
        "service": service,
        "amount": amount,
        "stage": "New",
        "priority": priority,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    st.session_state.leads = pd.concat(
        [st.session_state.leads, pd.DataFrame([new_lead])],
        ignore_index=True
    )


def update_lead_stage(lead_id, new_stage):
    st.session_state.leads.loc[
        st.session_state.leads["id"] == lead_id, "stage"
    ] = new_stage


def money(value):
    return f"₦{value:,.0f}"


# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Add Lead", "Priority Leads", "Pipeline Board", "Analytics Dashboard"])


# -----------------------------
# Page 1 — Add Lead
# -----------------------------
if page == "Add Lead":
    st.title("Add New Lead")

    name = st.text_input("Client Name")
    service = st.text_input("Service Type")
    amount = st.number_input("Service Estimate Amount (₦)", min_value=0)
    priority = st.selectbox("Lead Priority", ["Low", "Medium", "High"])

    if st.button("Submit Lead", type="primary", use_container_width=True):
        if name and service and amount:
            add_lead(name, service, amount, priority)
            st.success("Lead added successfully!")
        else:
            st.error("Please fill all fields.")

# -----------------------------
# Page 2 — Priority Leads
# -----------------------------
elif page == "Priority Leads":
    st.title("High Priority Leads")

    priority_df = st.session_state.leads[
        st.session_state.leads["priority"] == "High"
    ]

    if priority_df.empty:
        st.info("No high priority leads yet.")
    else:
        for _, row in priority_df.iterrows():
            st.markdown(
                f"""
                <div style='padding:12px; margin-bottom:10px; border-radius:10px; border:1px solid #ddd;'>
                    <b>{row['id']} — {row['name']} — {row['service']}</b>
                    <br>
                    <span style='color:green; font-size:20px; font-weight:bold;'>
                        {money(row['amount'])}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------
# Page 3 — Pipeline Board
# -----------------------------
elif page == "Pipeline Board":
    st.title("Sales Pipeline Board")

    stages = ["New", "Contacted", "Proposal Sent", "Negotiation", "Closed Won", "Closed Lost"]

    cols = st.columns(len(stages))

    for i, stage in enumerate(stages):
        with cols[i]:
            st.subheader(stage)

            stage_df = st.session_state.leads[
                st.session_state.leads["stage"] == stage
            ]

            if stage_df.empty:
                st.write("—")
            else:
                for _, lead in stage_df.iterrows():
                    st.markdown(
                        f"""
                        <div style='padding:10px; margin-bottom:12px; background:#f9f9f9;
                                    border-radius:10px; border:1px solid #ddd;'>
                            <b>{lead['id']} — {lead['name']}</b><br>
                            {lead['service']}<br>
                            <span style='color:#444; font-weight:bold;'>{money(lead['amount'])}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    new_stage = st.selectbox(
                        "Move to Stage",
                        stages,
                        index=stages.index(stage),
                        key=f"update_stage_{lead['id']}"
                    )

                    if st.button("Update", key=f"btn_update_{lead['id']}", use_container_width=True):
                        update_lead_stage(lead["id"], new_stage)
                        st.experimental_rerun()

            st.markdown("---")


# -----------------------------
# Page 4 — Analytics Dashboard
# -----------------------------
elif page == "Analytics Dashboard":
    st.title("Analytics Dashboard")

    df = st.session_state.leads

    if df.empty:
        st.info("No data yet.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Leads", len(df))

        with col2:
            st.metric("High Priority", len(df[df["priority"] == "High"]))

        with col3:
            total_amount = df["amount"].sum()
            st.metric("Total Pipeline Value", money(total_amount))

        st.subheader("Funnel Overview")

        funnel = df["stage"].value_counts().reindex(
            ["New", "Contacted", "Proposal Sent", "Negotiation", "Closed Won", "Closed Lost"],
            fill_value=0
        )

        st.bar_chart(funnel)

