# ==========================
# ✅ STEP 5: ML IMPLEMENTATION
# ==========================

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Make sure model directory exists
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pipeline_model.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ Feature Vector Builder
def build_feature_dataframe(leads):
    data = []
    for l in leads:
        data.append({
            "estimated_value": l.estimated_value or 0,
            "sla_hours": l.sla_hours or 48,
            "is_qualified": 1 if l.is_qualified else 0,
            "inspection_booked": 1 if l.status.name == "INSPECTION_SCHEDULED" else 0,
            "estimate_sent": 1 if l.status.name == "ESTIMATE_SUBMITTED" else 0,
            "status": l.status.value,
            "source": l.flags.split(",")[0] if l.flags else "unknown",
            "overdue": 1 if l.is_overdue else 0,
            "won": 1 if l.status.name == "AWARDED" else 0  # Target label
        })
    df = pd.DataFrame(data)

    # Encode categorical columns
    le_status = LabelEncoder()
    le_source = LabelEncoder()
    df["status_enc"] = le_status.fit_transform(df["status"])
    df["source_enc"] = le_source.fit_transform(df["source"])

    # Save encoders for later prediction
    joblib.dump({"status": le_status, "source": le_source}, os.path.join(MODEL_DIR, "encoders.joblib"))
    return df

# ✅ Model Trainer
def train_and_save_model(df):
    features = ["estimated_value", "sla_hours", "is_qualified", "inspection_booked", "estimate_sent", "status_enc", "source_enc", "overdue"]
    X = df[features]
    y = df["won"]

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "class_report": classification_report(y_test, y_pred),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    # Save Model
    joblib.dump({"model": model, "features": features, "metrics": metrics}, MODEL_PATH)
    return metrics

# ✅ Add training panel inside UI
if page == "Model Training":
    st.header("Lead Pipeline ML Model Trainer")
    st.markdown("*Train a baseline model on pipeline progression history and generate job-win probability signals.*")

    lead_df = build_feature_dataframe(leads)

    if st.button("🚀 Train & Save Model"):
        metrics = train_and_save_model(lead_df)

        st.success(f"Model Trained ✅  Accuracy: {metrics['accuracy']:.2f}, ROC-AUC: {metrics['roc_auc']:.2f}")
        st.text("Classification Report:")
        st.code(metrics["class_report"])
        st.text("Confusion Matrix:")
        st.write(metrics["conf_matrix"])

# ✅ Prediction Integration into pipeline dashboard
def load_model_bundle():
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["features"], bundle["metrics"]
    return None, None, None

model, features, metrics = load_model_bundle()

if model is not None and page == "Pipeline Board":
    st.markdown("### 🤖 ML Lead-Win Probability Surges")
    st.markdown("*AI-driven probability that lead converts into a job win based on pipeline strength and timing risk.*")

    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))

    for l in leads:
        vec = pd.DataFrame([{
            "estimated_value": l.estimated_value or 0,
            "sla_hours": l.sla_hours or 48,
            "is_qualified": 1 if l.is_qualified else 0,
            "inspection_booked": 1 if l.status.name == "INSPECTION_SCHEDULED" else 0,
            "estimate_sent": 1 if l.status.name == "ESTIMATE_SUBMITTED" else 0,
            "status_enc": encoders["status"].transform([l.status.value])[0],
            "source_enc": encoders["source"].transform([l.flags.split(",")[0] if l.flags else "unknown"])[0],
            "overdue": 1 if l.is_overdue else 0
        }])

        prob = model.predict_proba(vec[features])[:, 1][0]
        l.priority_score = round(l.priority_score + prob * 15, 2)  # priority upgrade merged with probability
        st.markdown(f"- {l.name} → Job-Win Probability: **{prob * 100:.1f}%** | Updated Score: {l.priority_score}")
