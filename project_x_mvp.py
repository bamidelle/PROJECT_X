# --- Page: Pipeline Board (Clean Rows layout — fully editable)
elif page == "Pipeline Board":
    st.header("🧭 Pipeline Board — Clean Rows")
    s = get_session()
    leads = s.query(Lead).order_by(Lead.created_at.desc()).all()

    if not leads:
        st.info("No leads yet. Create one from Lead Capture.")
    else:
        # Build DataFrame for priority scoring
        df = leads_df(s)
        weights = st.session_state.get("weights", {
            "value_weight": 0.5, "sla_weight": 0.35, "urgency_weight": 0.15,
            "contacted_w": 0.6, "inspection_w": 0.5, "estimate_w": 0.5, "value_baseline": 5000.0
        })

        priority_list = []
        for _, row in df.iterrows():
            score, _, _, _, _, _, time_left_h = compute_priority_for_lead_row(row, weights)
            priority_list.append({
                "id": int(row["id"]),
                "contact_name": row.get("contact_name") or "",
                "estimated_value": float(row.get("estimated_value") or 0.0),
                "time_left_hours": float(time_left_h),
                "priority_score": score,
                "status": row.get("status"),
            })
        pr_df = pd.DataFrame(priority_list).sort_values("priority_score", ascending=False)

        # Priority summary at top
        st.subheader("Priority Leads (Top 8)")
        if not pr_df.empty:
            for _, r in pr_df.head(8).iterrows():
                score = r["priority_score"]
                if score >= 0.7:
                    color = "red"
                elif score >= 0.45:
                    color = "orange"
                else:
                    color = "white"
                html_block = f"""
                <div style='padding:10px;border-radius:10px;margin-bottom:8px;border:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;justify-content:space-between;'>
                    <div>
                        <strong style='color:{color};'>#{int(r['id'])} — {r['contact_name'] or 'No name'}</strong>
                        <span style='color:var(--muted); margin-left:8px;'>| Est: ${r['estimated_value']:,.0f}</span>
                        <span style='color:var(--muted); margin-left:8px;'>| Time left: {int(r['time_left_hours'])}h</span>
                    </div>
                    <div style='font-weight:600; color:{color};'>Priority: {r['priority_score']:.2f}</div>
                </div>
                """
                st.markdown(html_block, unsafe_allow_html=True)
        else:
            st.info("No priority leads yet.")
        st.markdown("---")

        # Render leads as vertical rows (clean cards)
        for lead in leads:
            # Header line for each lead (compact)
            est_val_display = f"${lead.estimated_value:,.0f}" if lead.estimated_value else "$0"
            card_title = f"#{lead.id} — {lead.contact_name or 'No name'} — {lead.damage_type or 'No damage type'} — {est_val_display}"
            with st.expander(card_title, expanded=False):
                # Top info row
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**Source:** {lead.source or '—'}  &nbsp;&nbsp; **Assigned:** {lead.assigned_to or '—'}")
                    st.markdown(f"**Address:** {lead.property_address or '—'}")
                    st.markdown(f"**Notes:** {lead.notes or '—'}")
                    st.markdown(f"**Created:** {lead.created_at}")
                with col_b:
                    # compute and show priority for this specific lead
                    try:
                        single_row = df[df["id"] == lead.id].iloc[0].to_dict()
                        score, _, _, _, _, _, time_left_h = compute_priority_for_lead_row(single_row, weights)
                    except Exception:
                        score = 0.0
                        time_left_h = 9999
                    priority_label = ("High" if score >= 0.7 else "Medium" if score >= 0.45 else "Normal")
                    priority_color = "red" if score >= 0.7 else ("orange" if score >= 0.45 else "white")
                    st.markdown(f"<div style='text-align:right'><strong style='color:{priority_color};'>{priority_label}</strong><br><span style='color:var(--muted);'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)

                st.markdown("---")

                # Quick contact actions (Call / WhatsApp / Email)
                qc1, qc2, qc3, qc4 = st.columns([1,1,1,4])
                phone = (lead.contact_phone or "").strip()
                email = (lead.contact_email or "").strip()
                if phone:
                    tel_link = f"tel:{phone}"
                    wa_link = f"https://wa.me/{phone.lstrip('+').replace(' ', '')}?text=Hi%2C%20we%20are%20following%20up%20on%20your%20restoration%20request."
                    qc1.markdown(f"<a href='{tel_link}'><button>📞 Call</button></a>", unsafe_allow_html=True)
                    qc2.markdown(f"<a href='{wa_link}' target='_blank'><button>💬 WhatsApp</button></a>", unsafe_allow_html=True)
                else:
                    qc1.write(" ")
                    qc2.write(" ")
                if email:
                    mail_link = f"mailto:{email}?subject=Follow%20up%20on%20your%20restoration%20request"
                    qc3.markdown(f"<a href='{mail_link}'><button>✉️ Email</button></a>", unsafe_allow_html=True)
                else:
                    qc3.write(" ")
                qc4.write("")  # spacer column

                # SLA countdown display
                entered = lead.sla_entered_at or lead.created_at
                if isinstance(entered, str):
                    try:
                        entered = datetime.fromisoformat(entered)
                    except Exception:
                        entered = datetime.utcnow()
                deadline = entered + timedelta(hours=(lead.sla_hours or 24))
                remaining = deadline - datetime.utcnow()
                if remaining.total_seconds() <= 0:
                    st.markdown(f"❗ <strong style='color:red;'>SLA OVERDUE</strong> — was due {deadline.strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                else:
                    st.markdown(f"⏳ SLA remaining: {str(remaining).split('.')[0]} (due {deadline.strftime('%Y-%m-%d %H:%M')})")

                st.markdown("---")

                # Editable form for this lead (NO st.rerun inside)
                form_key = f"lead_edit_form_{lead.id}"
                with st.form(form_key):
                    c1, c2 = st.columns(2)
                    with c1:
                        contact_name = st.text_input("Contact name", value=lead.contact_name or "", key=f"cname_{lead.id}")
                        contact_phone = st.text_input("Contact phone", value=lead.contact_phone or "", key=f"cphone_{lead.id}")
                        contact_email = st.text_input("Contact email", value=lead.contact_email or "", key=f"cemail_{lead.id}")
                        property_address = st.text_input("Property address", value=lead.property_address or "", key=f"addr_{lead.id}")
                        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"], index=(["water","fire","mold","contents","reconstruction","other"].index(lead.damage_type) if lead.damage_type in ["water","fire","mold","contents","reconstruction","other"] else 5), key=f"damage_{lead.id}")
                    with c2:
                        assigned_to = st.text_input("Assigned to", value=lead.assigned_to or "", key=f"assign_{lead.id}")
                        est_val = st.number_input("Estimated value (USD)", min_value=0.0, value=float(lead.estimated_value or 0.0), step=50.0, key=f"est_{lead.id}")
                        sla_hours = st.number_input("SLA hours", min_value=1, value=int(lead.sla_hours or 24), step=1, key=f"sla_{lead.id}")
                        status_choice = st.selectbox("Status", options=LeadStatus.ALL, index=LeadStatus.ALL.index(lead.status), key=f"status_{lead.id}")

                    notes = st.text_area("Notes", value=lead.notes or "", key=f"notes_{lead.id}")

                    # Pipeline flags & dates
                    st.markdown("**Pipeline Steps**")
                    colf1, colf2, colf3 = st.columns(3)
                    with colf1:
                        contacted_choice = st.selectbox("Contacted?", ["No", "Yes"], index=1 if lead.contacted else 0, key=f"cont_{lead.id}")
                        inspection_scheduled_choice = st.selectbox("Inspection Scheduled?", ["No", "Yes"], index=1 if lead.inspection_scheduled else 0, key=f"inspsch_{lead.id}")
                        if inspection_scheduled_choice == "Yes":
                            default_dt = lead.inspection_scheduled_at or datetime.utcnow()
                            if isinstance(default_dt, str):
                                try:
                                    default_dt = datetime.fromisoformat(default_dt)
                                except Exception:
                                    default_dt = datetime.utcnow()
                            inspection_dt = st.datetime_input("Inspection date & time", value=default_dt, key=f"insp_dt_{lead.id}")
                        else:
                            inspection_dt = None
                    with colf2:
                        inspection_completed_choice = st.selectbox("Inspection Completed?", ["No", "Yes"], index=1 if lead.inspection_completed else 0, key=f"inspcomp_{lead.id}")
                        if inspection_completed_choice == "Yes":
                            default_dt2 = lead.inspection_completed_at or datetime.utcnow()
                            if isinstance(default_dt2, str):
                                try:
                                    default_dt2 = datetime.fromisoformat(default_dt2)
                                except Exception:
                                    default_dt2 = datetime.utcnow()
                            inspection_comp_dt = st.datetime_input("Inspection completed at", value=default_dt2, key=f"insp_comp_dt_{lead.id}")
                        else:
                            inspection_comp_dt = None
                        estimate_sub_choice = st.selectbox("Estimate Submitted?", ["No","Yes"], index=1 if lead.estimate_submitted else 0, key=f"estsub_{lead.id}")
                        if estimate_sub_choice == "Yes":
                            est_submitted_at = st.date_input("Estimate submitted at (optional)", value=(lead.estimate_submitted_at.date() if lead.estimate_submitted_at else datetime.utcnow().date()), key=f"est_sub_dt_{lead.id}")
                        else:
                            est_submitted_at = None
                    with colf3:
                        awarded_comment = st.text_input("Awarded comment (optional)", value=lead.awarded_comment or "", key=f"awcom_{lead.id}")
                        awarded_date = st.date_input("Awarded date (optional)", value=(lead.awarded_date.date() if lead.awarded_date else datetime.utcnow().date()), key=f"awdate_{lead.id}")
                        lost_comment = st.text_input("Lost comment (optional)", value=lead.lost_comment or "", key=f"lostcom_{lead.id}")
                        lost_date = st.date_input("Lost date (optional)", value=(lead.lost_date.date() if lead.lost_date else datetime.utcnow().date()), key=f"lostdate_{lead.id}")

                    # Save button
                    save_btn = st.form_submit_button("Save Lead")
                    if save_btn:
                        try:
                            # Apply changes to SQLAlchemy object
                            lead.contact_name = contact_name.strip() or None
                            lead.contact_phone = contact_phone.strip() or None
                            lead.contact_email = contact_email.strip() or None
                            lead.property_address = property_address.strip() or None
                            lead.damage_type = damage_type
                            lead.assigned_to = assigned_to.strip() or None
                            lead.estimated_value = float(est_val) if est_val else None
                            lead.notes = notes.strip() or None
                            lead.sla_hours = int(sla_hours)
                            # pipeline flags
                            lead.contacted = True if contacted_choice == "Yes" else False
                            lead.inspection_scheduled = True if inspection_scheduled_choice == "Yes" else False
                            lead.inspection_scheduled_at = inspection_dt
                            lead.inspection_completed = True if inspection_completed_choice == "Yes" else False
                            lead.inspection_completed_at = inspection_comp_dt
                            lead.estimate_submitted = True if estimate_sub_choice == "Yes" else False
                            lead.estimate_submitted_at = (datetime.combine(est_submitted_at, datetime.min.time()) if est_submitted_at else None)
                            lead.awarded_comment = awarded_comment.strip() or None
                            lead.awarded_date = (datetime.combine(awarded_date, datetime.min.time()) if awarded_comment or awarded_date else None)
                            lead.lost_comment = lost_comment.strip() or None
                            lead.lost_date = (datetime.combine(lost_date, datetime.min.time()) if lost_comment or lost_date else None)
                            # status change
                            if status_choice != lead.status:
                                lead.status = status_choice
                                lead.sla_stage = status_choice
                                lead.sla_entered_at = datetime.utcnow()
                            s.add(lead)
                            s.commit()
                            st.success(f"Lead #{lead.id} saved.")
                        except Exception as e:
                            st.error(f"Error saving lead #{lead.id}: {e}")

                # Estimates section (outside the edit form to avoid nested form issues)
                st.markdown("**Estimates**")
                ests = s.query(Estimate).filter(Estimate.lead_id == lead.id).order_by(Estimate.created_at.desc()).all()
                if ests:
                    est_rows = []
                    for e in ests:
                        est_rows.append({
                            "id": e.id,
                            "amount": e.amount,
                            "sent_at": e.sent_at,
                            "approved": e.approved,
                            "lost": e.lost,
                            "lost_reason": e.lost_reason,
                            "created_at": e.created_at
                        })
                    st.dataframe(pd.DataFrame(est_rows))
                    # actions for first estimate
                    first_est = ests[0]
                    ea, eb, ec = st.columns(3)
                    with ea:
                        if st.button(f"Mark Sent (#{first_est.id})", key=f"send_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_sent(s, first_est.id)
                                st.success("Marked as sent.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with eb:
                        if st.button(f"Mark Approved (#{first_est.id})", key=f"app_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_approved(s, first_est.id)
                                st.success("Marked approved; lead moved to Awarded.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with ec:
                        if st.button(f"Mark Lost (#{first_est.id})", key=f"lost_{lead.id}_{first_est.id}"):
                            try:
                                mark_estimate_lost(s, first_est.id, reason="Lost to competitor")
                                st.success("Marked lost; lead moved to Lost.")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                else:
                    st.write("No estimates yet.")
                    # create estimate form
                    with st.form(f"create_est_{lead.id}", clear_on_submit=True):
                        amt = st.number_input("Estimate amount (USD)", min_value=0.0, value=lead.estimated_value or 0.0, step=50.0, key=f"new_est_amt_{lead.id}")
                        det = st.text_area("Estimate details (optional)", key=f"new_est_det_{lead.id}")
                        create_btn = st.form_submit_button("Create Estimate")
                        if create_btn:
                            try:
                                create_estimate(s, lead.id, float(amt), details=det)
                                st.success("Estimate created.")
                            except Exception as e:
                                st.error(f"Failed to create estimate: {e}")

                st.markdown("---")
