# models.py
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()
DB_FILE = "leads.db"
engine = create_engine(f"sqlite:///{DB_FILE}")
SessionLocal = sessionmaker(bind=engine)

class LeadStatus:
    NEW = "New"
    CONTACTED = "Contacted"
    INSPECTION = "Inspection Scheduled"
    ESTIMATE_SUBMITTED = "Estimate Submitted"
    AWARDED = "Awarded"
    LOST = "Lost"
    ALL = [NEW, CONTACTED, INSPECTION, ESTIMATE_SUBMITTED, AWARDED, LOST]

class Lead(Base):
    __tablename__ = "leads"
    id = Column(Integer, primary_key=True)
    contact_name = Column(String)
    contact_phone = Column(String)
    contact_email = Column(String)
    property_address = Column(String)
    damage_type = Column(String)
    assigned_to = Column(String)
    estimated_value = Column(Float)
    notes = Column(String)
    status = Column(String, default=LeadStatus.NEW)
    sla_hours = Column(Integer, default=24)
    sla_entered_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    qualified = Column(Boolean, default=True)
    # Add other fields like inspection_scheduled, estimate_submitted, etc.

class Estimate(Base):
    __tablename__ = "estimates"
    id = Column(Integer, primary_key=True)
    lead_id = Column(Integer)
    amount = Column(Float)
    sent_at = Column(DateTime)
    approved = Column(Boolean, default=False)
    lost = Column(Boolean, default=False)
    lost_reason = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(engine)

def get_session():
    return SessionLocal()

def leads_df(session):
    leads = session.query(Lead).all()
    data = []
    for l in leads:
        data.append({
            "id": l.id,
            "contact_name": l.contact_name,
            "contact_phone": l.contact_phone,
            "contact_email": l.contact_email,
            "property_address": l.property_address,
            "damage_type": l.damage_type,
            "assigned_to": l.assigned_to,
            "estimated_value": l.estimated_value,
            "notes": l.notes,
            "status": l.status,
            "sla_hours": l.sla_hours,
            "sla_entered_at": l.sla_entered_at,
            "created_at": l.created_at,
            "qualified": l.qualified
        })
    import pandas as pd
    return pd.DataFrame(data)

def estimates_df(session):
    estimates = session.query(Estimate).all()
    data = []
    for e in estimates:
        data.append({
            "id": e.id,
            "lead_id": e.lead_id,
            "amount": e.amount,
            "sent_at": e.sent_at,
            "approved": e.approved,
            "lost": e.lost,
            "lost_reason": e.lost_reason,
            "created_at": e.created_at
        })
    import pandas as pd
    return pd.DataFrame(data)

# Add simple helper functions
def add_lead(session, **kwargs):
    lead = Lead(**kwargs)
    session.add(lead)
    session.commit()
    session.refresh(lead)
    return lead

def create_estimate(session, lead_id, amount, details=None):
    est = Estimate(lead_id=lead_id, amount=amount)
    session.add(est)
    session.commit()
    return est

def mark_estimate_sent(session, est_id):
    est = session.query(Estimate).get(est_id)
    est.sent_at = datetime.utcnow()
    session.commit()

def mark_estimate_approved(session, est_id):
    est = session.query(Estimate).get(est_id)
    est.approved = True
    # Move lead to Awarded
    lead = session.query(Lead).get(est.lead_id)
    lead.status = LeadStatus.AWARDED
    session.commit()

def mark_estimate_lost(session, est_id, reason=None):
    est = session.query(Estimate).get(est_id)
    est.lost = True
    est.lost_reason = reason
    lead = session.query(Lead).get(est.lead_id)
    lead.status = LeadStatus.LOST
    session.commit()
