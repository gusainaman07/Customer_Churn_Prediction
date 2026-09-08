from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f7f9fc 0%, #edf5ff 50%, #f5f3ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #1f2d3d;
            font-weight: 700;
            letter-spacing: -0.04em;
        }
        .metric-card {
            background: rgba(255,255,255,0.8);
            border: 1px solid rgba(31,45,61,0.08);
            border-radius: 18px;
            padding: 1rem 1.25rem;
            box-shadow: 0 8px 28px rgba(31, 45, 61, 0.08);
        }
        .section-title {
            color: #1f2d3d;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .pill {
            display: inline-block;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        .sidebar .block-container {
            padding-top: 1.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Customer Churn Predictor")
st.caption("Estimate a customer's likelihood of churn using the saved retention model.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_band(probability):
    if probability < 0.30:
        return "Low Risk", "✅", "#22c55e"
    if probability < 0.70:
        return "Medium Risk", "⚠️", "#f59e0b"
    return "High Risk", "🚨", "#ef4444"


model = load_model()

with st.sidebar:
    st.header("Customer profile")
    with st.form("customer_form", clear_on_submit=False):
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.subheader("Account details")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly charges", min_value=0.0, max_value=150.0, value=70.0
        )
        total_charges = st.number_input(
            "Total charges", min_value=0.0, max_value=10000.0, value=840.0
        )

        st.subheader("Services")
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming movies", ["Yes", "No", "No internet service"]
        )

        submitted = st.form_submit_button("Predict churn", use_container_width=True)

if submitted:
    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
    ]

    if tenure <= 6:
        tenure_group = "0-6 months"
    elif tenure <= 12:
        tenure_group = "7-12 months"
    elif tenure <= 24:
        tenure_group = "13-24 months"
    elif tenure <= 48:
        tenure_group = "25-48 months"
    else:
        tenure_group = "49-72 months"

    customer = pd.DataFrame([
        {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "services_count": sum(value == "Yes" for value in service_values),
            "tenure_group": tenure_group,
            "has_security_or_support": int(
                online_security == "Yes" or tech_support == "Yes"
            ),
        }
    ])

    probability = model.predict_proba(customer)[0, 1]
    risk, emoji, color = risk_band(probability)

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-card"><div style="font-size: 0.9rem; color: #5f6f85;">Churn probability</div><div style="font-size: 2.4rem; font-weight: 800; color: #1f2d3d; margin-top: 0.35rem;">{probability:.1%}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="pill" style="background: {color}22; color: {color}; margin-top: 1rem;">{emoji} {risk}</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(100, max(0, int(probability * 100))))

    with right:
        st.markdown('<div class="section-title">Retention summary</div>', unsafe_allow_html=True)
        if risk == "Low Risk":
            st.success("This customer is likely to stay and may not need immediate retention outreach.")
        elif risk == "Medium Risk":
            st.warning("This customer may deserve a targeted retention offer based on service and tenure profile.")
        else:
            st.error("This customer shows a high churn likelihood and should be prioritized for retention action.")

        st.caption(
            "The score combines contract type, tenure, service usage, support coverage, and billing behavior."
        )

else:
    st.info("Fill out the customer information in the sidebar to generate a churn-risk prediction.")
