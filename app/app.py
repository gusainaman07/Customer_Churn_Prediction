from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"

st.set_page_config(page_title="Customer Churn Predictor")
st.title("Customer Churn Predictor")
st.write("Estimate a customer's probability of churn.")


@st.cache_resource
def load_model():
	return joblib.load(MODEL_PATH)


model = load_model()

with st.form("customer_form"):
	gender = st.selectbox("Gender", ["Female", "Male"])
	senior_citizen = st.selectbox("Senior citizen", [0, 1])
	partner = st.selectbox("Partner", ["Yes", "No"])
	dependents = st.selectbox("Dependents", ["Yes", "No"])
	tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
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

	submitted = st.form_submit_button("Predict churn")


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

	customer = pd.DataFrame([{
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
	}])

	probability = model.predict_proba(customer)[0, 1]
	risk = "Low Risk" if probability < 0.30 else "Medium Risk" if probability < 0.70 else "High Risk"

	st.subheader("Prediction")
	st.metric("Churn probability", f"{probability:.1%}")
	if risk == "Low Risk":
		st.success(risk)
	else:
		st.warning(risk)
