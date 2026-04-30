import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="SRGL/machine-failure-prediction", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tour Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase a tour package
based on their profile and interaction details.
Please enter the customer information below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18.0, max_value=61.0, value=36.0, step=1.0)

typeof_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Enquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_of_pitch = st.number_input(
    "Duration of Pitch",
    min_value=5.0,
    max_value=127.0,
    value=14.0,
    step=1.0
)

occupation = st.selectbox(
    "Occupation",
    ["Free Lancer", "Large Business", "Salaried", "Small Business"]
)

gender = st.selectbox(
    "Gender",
    ["Fe Male", "Female", "Male"]
)

number_of_person_visiting = st.number_input(
    "Number of Person Visiting",
    min_value=1,
    max_value=5,
    value=3
)

number_of_followups = st.number_input(
    "Number of Followups",
    min_value=1.0,
    max_value=6.0,
    value=4.0,
    step=1.0
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3.0, 4.0, 5.0]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Divorced", "Married", "Single", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips",
    min_value=1.0,
    max_value=22.0,
    value=3.0,
    step=1.0
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score",
    [1, 2, 3, 4, 5]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.selectbox(
    "Number of Children Visiting",
    [0.0, 1.0, 2.0, 3.0]
)

designation = st.selectbox(
    "Designation",
    ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000.0,
    max_value=98678.0,
    value=22421.0,
    step=1000.0
)

input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [typeof_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

if st.button("Predict Customer Response"):
    prediction = model.predict(input_data)[0]

    result = "Customer Will Take the Product" if prediction == 1 else "Customer Will Not Take the Product"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
