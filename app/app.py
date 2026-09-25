from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"
SCORED_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "churn_scored.csv"

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
            padding-top: 2.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }
        h1 {
            color: #1f2d3d;
            font-weight: 700;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }
        .intro {
            color: #5f6f85;
            font-size: 1.05rem;
            margin-bottom: 1.4rem;
        }
        .guide-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(31, 45, 61, 0.09);
            border-radius: 14px;
            padding: 1.25rem 1.35rem;
            min-height: 235px;
            box-shadow: 0 8px 24px rgba(31, 45, 61, 0.06);
        }
        .guide-card h3 {
            color: #1f2d3d;
            font-size: 1.08rem;
            margin: 0 0 0.7rem;
        }
        .guide-card ul,
        .guide-card ol {
            color: #4f6075;
            line-height: 1.55;
            margin: 0;
            padding-left: 1.15rem;
        }
        .guide-card li + li {
            margin-top: 0.45rem;
        }
        .guide-label {
            color: #2764a5;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }
        .result-guide {
            background: rgba(255, 255, 255, 0.62);
            border-left: 4px solid #4f8ac9;
            border-radius: 8px;
            color: #4f6075;
            line-height: 1.5;
            margin: 1.25rem 0 1.5rem;
            padding: 0.9rem 1rem;
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
st.markdown(
    '<div class="intro">A simple telecom retention tool that turns customer details into an actionable churn-risk estimate.</div>',
    unsafe_allow_html=True,
)

@st.cache_data
def load_scored_data():
    return pd.read_csv(SCORED_DATA_PATH)


overview_tab, usage_tab, output_tab, risk_table_tab = st.tabs(
    ["About this project", "How to use it", "Understand the output", "Customer risk table"]
)

with overview_tab:
    st.subheader("What this project is about")
    st.markdown(
        """
        - Predicts whether a telecom customer may leave the company.
        - Uses customer demographics, contract, tenure, services, and billing details.
        - Helps retention teams prioritize customers who may need attention.
        - The model was trained on historical telecom customer data.
        """
    )

with usage_tab:
    st.subheader("How to use the app")
    st.markdown(
        """
        1. Complete the customer profile in the left sidebar.
        2. Enter realistic account, service, and billing information.
        3. Select **Predict churn** at the bottom of the sidebar.
        4. Review the probability, risk category, and retention summary.
        """
    )

with output_tab:
    st.subheader("What the output means")
    output_columns = st.columns(3, gap="medium")
    output_columns[0].success("**Low Risk**\n\nBelow 30%, likely to stay.")
    output_columns[1].warning("**Medium Risk**\n\n30% to below 70%, may benefit from a targeted offer.")
    output_columns[2].error("**High Risk**\n\n70% or more, prioritize for retention outreach.")
    st.caption("The percentage is an estimate, not a guarantee that a customer will leave.")

with risk_table_tab:
    st.subheader("Customer risk table")
    st.write(
        "Use the interactive table to sort, filter, and review customers from the scored dataset."
    )
    risk_filter = st.selectbox(
        "Show customers in this risk category",
        ["All risks", "High Risk", "Medium Risk", "Low Risk"],
    )
    scored_data = load_scored_data()
    if risk_filter != "All risks":
        scored_data = scored_data[scored_data["risk_category"] == risk_filter]

    grid_data = scored_data[
        [
            "customerID",
            "tenure",
            "Contract",
            "InternetService",
            "MonthlyCharges",
            "churn_probability",
            "risk_category",
        ]
    ].copy()
    grid_data["churn_probability"] = grid_data["churn_probability"].round(4)
    grid_data = grid_data.sort_values("churn_probability", ascending=False)

    grid_options = GridOptionsBuilder.from_dataframe(grid_data)
    grid_options.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
    )
    grid_options.configure_pagination(
        paginationAutoPageSize=False,
        paginationPageSize=15,
    )
    grid_options.configure_column(
        "churn_probability",
        header_name="Churn probability",
        type=["numericColumn", "numberColumnFilter"],
    )
    grid_options.configure_side_bar()

    AgGrid(
        grid_data,
        gridOptions=grid_options.build(),
        height=480,
        theme="streamlit",
        fit_columns_on_grid_load=True,
    )

st.markdown(
    '<div class="result-guide"><strong>Tip:</strong> Try different customer profiles to see which combinations of contract, tenure, services, and charges increase or reduce churn risk.</div>',
    unsafe_allow_html=True,
)


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

    st.divider()
    left, right = st.columns([1.5, 1])

    with left:
        st.subheader("Prediction")
        metric_columns = st.columns(2)
        metric_columns[0].metric("Churn probability", f"{probability:.1%}")
        metric_columns[1].metric("Risk category", f"{emoji} {risk}")
        st.progress(
            min(100, max(0, int(probability * 100))),
            text="Estimated churn probability",
        )

    with right:
        st.subheader("Retention summary")
        if risk == "Low Risk":
            st.success("This customer is likely to stay and may not need immediate retention outreach.")
        elif risk == "Medium Risk":
            st.warning("This customer may deserve a targeted retention offer based on service and tenure profile.")
        else:
            st.error("This customer shows a high churn likelihood and should be prioritized for retention action.")

        st.caption(
            "The score combines contract type, tenure, service usage, support coverage, and billing behavior."
        )

    with st.expander("View customer profile used for this prediction"):
        profile_summary = pd.DataFrame(
            {
                "Category": ["Contract", "Tenure", "Monthly charges", "Internet service", "Payment method", "Services selected"],
                "Value": [
                    contract,
                    f"{tenure:.0f} months",
                    f"${monthly_charges:,.2f}",
                    internet_service,
                    payment_method,
                    str(customer.at[0, "services_count"]),
                ],
            }
        )
        st.dataframe(profile_summary, hide_index=True, use_container_width=True)

else:
    st.info("Fill out the customer information in the sidebar to generate a churn-risk prediction.")
