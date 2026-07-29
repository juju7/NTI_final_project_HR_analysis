import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


model, scaler = load_model()

CATEGORY_VALUES = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": ["Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"],
    "Gender": ["Female", "Male"],
    "JobRole": ["Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
                "Manufacturing Director", "Research Director", "Research Scientist",
                "Sales Executive", "Sales Representative"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

encoders = {col: LabelEncoder().fit(vals) for col, vals in CATEGORY_VALUES.items()}

features = [
    "Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome",
    "Education", "EducationField", "EmployeeNumber", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
]

st.title("👔 Employee Attrition Predictor")
st.write("Enter an employee's details below to estimate their risk of leaving the company.")

if "step" not in st.session_state:
    st.session_state.step = 1

DEFAULTS = {
    "age": 30, "gender": encoders["Gender"].classes_[0],
    "marital_status": encoders["MaritalStatus"].classes_[0],
    "education": 3, "education_field": encoders["EducationField"].classes_[0],
    "distance_from_home": 5,
    "department": encoders["Department"].classes_[0],
    "job_role": encoders["JobRole"].classes_[0],
    "job_level": 2, "business_travel": encoders["BusinessTravel"].classes_[0],
    "overtime": "No", "employee_number": 1,
    "job_involvement": 3, "performance_rating": 3,
    "monthly_income": 5000, "monthly_rate": 14000,
    "daily_rate": 800, "hourly_rate": 65,
    "percent_salary_hike": 15, "stock_option_level": 1,
    "job_satisfaction": 3, "env_satisfaction": 3,
    "relationship_satisfaction": 3, "work_life_balance": 3,
    "total_working_years": 8, "years_at_company": 5,
    "years_in_current_role": 3, "years_since_promotion": 1,
    "years_with_curr_manager": 3, "num_companies_worked": 2,
    "training_times_last_year": 2,
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.progress(st.session_state.step / 4)

titles = [
    "👤 Personal",
    "💼 Job Info",
    "💰 Compensation",
    "📈 Satisfaction & Tenure"
]

st.subheader(f"Step {st.session_state.step}/4: {titles[st.session_state.step - 1]}")

with st.form("employee_form"):

    # ---------------- STEP 1 ----------------
    if st.session_state.step == 1:

        st.slider("Age", 18, 60, 30, key="age")
        st.selectbox("Gender", encoders["Gender"].classes_, key="gender")
        st.selectbox("Marital Status", encoders["MaritalStatus"].classes_, key="marital_status")
        st.slider("Education (1=Below College ... 5=Doctor)", 1, 5, 3, key="education")
        st.selectbox("Education Field", encoders["EducationField"].classes_, key="education_field")
        st.slider("Distance From Home (miles)", 1, 30, 5, key="distance_from_home")

        next1 = st.form_submit_button("Next ➜")
        back = submitted = False

        if next1:
            st.session_state.step = 2
            st.rerun()

    # ---------------- STEP 2 ----------------
    elif st.session_state.step == 2:

        st.selectbox("Department", encoders["Department"].classes_, key="department")
        st.selectbox("Job Role", encoders["JobRole"].classes_, key="job_role")
        st.slider("Job Level", 1, 5, 2, key="job_level")
        st.selectbox("Business Travel", encoders["BusinessTravel"].classes_, key="business_travel")
        st.selectbox("OverTime", ["Yes", "No"], key="overtime")
        st.number_input("Employee Number", 1, 3000, 1, key="employee_number")
        st.slider("Job Involvement", 1, 4, 3, key="job_involvement")
        st.slider("Performance Rating", 1, 4, 3, key="performance_rating")

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            next2 = st.form_submit_button("Next ➜")
        submitted = False

        if back:
            st.session_state.step = 1
            st.rerun()
        if next2:
            st.session_state.step = 3
            st.rerun()

    # ---------------- STEP 3 ----------------
    elif st.session_state.step == 3:

        st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=100, key="monthly_income")
        st.number_input("Monthly Rate", 2000, 27000, 14000, step=100, key="monthly_rate")
        st.number_input("Daily Rate", 100, 1500, 800, step=10, key="daily_rate")
        st.number_input("Hourly Rate", 30, 100, 65, key="hourly_rate")
        st.slider("Percent Salary Hike", 11, 25, 15, key="percent_salary_hike")
        st.slider("Stock Option Level", 0, 3, 1, key="stock_option_level")

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            next3 = st.form_submit_button("Next ➜")
        submitted = False

        if back:
            st.session_state.step = 2
            st.rerun()
        if next3:
            st.session_state.step = 4
            st.rerun()

    # ---------------- STEP 4 ----------------
    else:

        st.slider("Job Satisfaction", 1, 4, 3, key="job_satisfaction")
        st.slider("Environment Satisfaction", 1, 4, 3, key="env_satisfaction")
        st.slider("Relationship Satisfaction", 1, 4, 3, key="relationship_satisfaction")
        st.slider("Work-Life Balance", 1, 4, 3, key="work_life_balance")
        st.slider("Total Working Years", 0, 40, 8, key="total_working_years")
        st.slider("Years at Company", 0, 40, 5, key="years_at_company")
        st.slider("Years in Current Role", 0, 20, 3, key="years_in_current_role")
        st.slider("Years Since Last Promotion", 0, 15, 1, key="years_since_promotion")
        st.slider("Years With Current Manager", 0, 20, 3, key="years_with_curr_manager")
        st.slider("Number of Companies Worked At", 0, 10, 2, key="num_companies_worked")
        st.slider("Training Times Last Year", 0, 6, 2, key="training_times_last_year")

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            submitted = st.form_submit_button("🚀 Predict Attrition Risk")

        if back:
            st.session_state.step = 3
            st.rerun()


if st.session_state.step == 4 and submitted:
    missing = [k for k in DEFAULTS if k not in st.session_state]
    if missing:
        st.warning("Please complete all steps before predicting.")
        st.stop()

    s = st.session_state  # shorthand

    input_dict = {
        "Age": s.age,
        "BusinessTravel": encoders["BusinessTravel"].transform([s.business_travel])[0],
        "DailyRate": s.daily_rate,
        "Department": encoders["Department"].transform([s.department])[0],
        "DistanceFromHome": s.distance_from_home,
        "Education": s.education,
        "EducationField": encoders["EducationField"].transform([s.education_field])[0],
        "EmployeeNumber": s.employee_number,
        "EnvironmentSatisfaction": s.env_satisfaction,
        "Gender": encoders["Gender"].transform([s.gender])[0],
        "HourlyRate": s.hourly_rate,
        "JobInvolvement": s.job_involvement,
        "JobLevel": s.job_level,
        "JobRole": encoders["JobRole"].transform([s.job_role])[0],
        "JobSatisfaction": s.job_satisfaction,
        "MaritalStatus": encoders["MaritalStatus"].transform([s.marital_status])[0],
        "MonthlyIncome": s.monthly_income,
        "MonthlyRate": s.monthly_rate,
        "NumCompaniesWorked": s.num_companies_worked,
        "OverTime": encoders["OverTime"].transform([s.overtime])[0],
        "PercentSalaryHike": s.percent_salary_hike,
        "PerformanceRating": s.performance_rating,
        "RelationshipSatisfaction": s.relationship_satisfaction,
        "StockOptionLevel": s.stock_option_level,
        "TotalWorkingYears": s.total_working_years,
        "TrainingTimesLastYear": s.training_times_last_year,
        "WorkLifeBalance": s.work_life_balance,
        "YearsAtCompany": s.years_at_company,
        "YearsInCurrentRole": s.years_in_current_role,
        "YearsSinceLastPromotion": s.years_since_promotion,
        "YearsWithCurrManager": s.years_with_curr_manager,
    }

    input_df = pd.DataFrame([input_dict])[features]

    # Apply the same StandardScaler used during training
    input_scaled = scaler.transform(input_df.values)

   
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0][1]
    else:
        raw_output = model.predict(input_scaled)
        proba = float(np.array(raw_output).ravel()[0])

    prediction = "Likely to Leave ⚠️" if proba >= 0.5 else "Likely to Stay ✅"

    st.subheader("Prediction Result")
    st.metric("Attrition Probability", f"{proba*100:.1f}%")

    if proba >= 0.5:
        st.error(prediction)
    else:
        st.success(prediction)

    st.progress(min(int(proba * 100), 100))

    with st.expander("See input summary"):
        st.write(input_dict)