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

# Persistent store for answers across steps + which step we're on
if "step" not in st.session_state:
    st.session_state.step = 1
if "answers" not in st.session_state:
    st.session_state.answers = {}

a = st.session_state.answers  # shorthand - a plain dict we control ourselves

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

        age = st.slider("Age", 18, 60, a.get("age", 30))
        gender = st.selectbox("Gender", encoders["Gender"].classes_,
                               index=list(encoders["Gender"].classes_).index(a["gender"]) if "gender" in a else 0)
        marital_status = st.selectbox(
            "Marital Status", encoders["MaritalStatus"].classes_,
            index=list(encoders["MaritalStatus"].classes_).index(a["marital_status"]) if "marital_status" in a else 0)
        education = st.slider("Education (1=Below College ... 5=Doctor)", 1, 5, a.get("education", 3))
        education_field = st.selectbox(
            "Education Field", encoders["EducationField"].classes_,
            index=list(encoders["EducationField"].classes_).index(a["education_field"]) if "education_field" in a else 0)
        distance_from_home = st.slider("Distance From Home (miles)", 1, 30, a.get("distance_from_home", 5))

        next1 = st.form_submit_button("Next ➜")

        if next1:
            a.update({
                "age": age, "gender": gender, "marital_status": marital_status,
                "education": education, "education_field": education_field,
                "distance_from_home": distance_from_home,
            })
            st.session_state.step = 2
            st.rerun()

    # ---------------- STEP 2 ----------------
    elif st.session_state.step == 2:

        department = st.selectbox(
            "Department", encoders["Department"].classes_,
            index=list(encoders["Department"].classes_).index(a["department"]) if "department" in a else 0)
        job_role = st.selectbox(
            "Job Role", encoders["JobRole"].classes_,
            index=list(encoders["JobRole"].classes_).index(a["job_role"]) if "job_role" in a else 0)
        job_level = st.slider("Job Level", 1, 5, a.get("job_level", 2))
        business_travel = st.selectbox(
            "Business Travel", encoders["BusinessTravel"].classes_,
            index=list(encoders["BusinessTravel"].classes_).index(a["business_travel"]) if "business_travel" in a else 0)
        overtime = st.selectbox("OverTime", ["Yes", "No"],
                                 index=["Yes", "No"].index(a.get("overtime", "No")))
        employee_number = st.number_input("Employee Number", 1, 3000, a.get("employee_number", 1))
        job_involvement = st.slider("Job Involvement", 1, 4, a.get("job_involvement", 3))
        performance_rating = st.slider("Performance Rating", 1, 4, a.get("performance_rating", 3))

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            next2 = st.form_submit_button("Next ➜")

        if back or next2:
            a.update({
                "department": department, "job_role": job_role, "job_level": job_level,
                "business_travel": business_travel, "overtime": overtime,
                "employee_number": employee_number, "job_involvement": job_involvement,
                "performance_rating": performance_rating,
            })
            st.session_state.step = 1 if back else 3
            st.rerun()

    # ---------------- STEP 3 ----------------
    elif st.session_state.step == 3:

        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, a.get("monthly_income", 5000), step=100)
        monthly_rate = st.number_input("Monthly Rate", 2000, 27000, a.get("monthly_rate", 14000), step=100)
        daily_rate = st.number_input("Daily Rate", 100, 1500, a.get("daily_rate", 800), step=10)
        hourly_rate = st.number_input("Hourly Rate", 30, 100, a.get("hourly_rate", 65))
        percent_salary_hike = st.slider("Percent Salary Hike", 11, 25, a.get("percent_salary_hike", 15))
        stock_option_level = st.slider("Stock Option Level", 0, 3, a.get("stock_option_level", 1))

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            next3 = st.form_submit_button("Next ➜")

        if back or next3:
            a.update({
                "monthly_income": monthly_income, "monthly_rate": monthly_rate,
                "daily_rate": daily_rate, "hourly_rate": hourly_rate,
                "percent_salary_hike": percent_salary_hike, "stock_option_level": stock_option_level,
            })
            st.session_state.step = 2 if back else 4
            st.rerun()

    # ---------------- STEP 4 ----------------
    else:

        job_satisfaction = st.slider("Job Satisfaction", 1, 4, a.get("job_satisfaction", 3))
        env_satisfaction = st.slider("Environment Satisfaction", 1, 4, a.get("env_satisfaction", 3))
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, a.get("relationship_satisfaction", 3))
        work_life_balance = st.slider("Work-Life Balance", 1, 4, a.get("work_life_balance", 3))
        total_working_years = st.slider("Total Working Years", 0, 40, a.get("total_working_years", 8))
        years_at_company = st.slider("Years at Company", 0, 40, a.get("years_at_company", 5))
        years_in_current_role = st.slider("Years in Current Role", 0, 20, a.get("years_in_current_role", 3))
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, a.get("years_since_promotion", 1))
        years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, a.get("years_with_curr_manager", 3))
        num_companies_worked = st.slider("Number of Companies Worked At", 0, 10, a.get("num_companies_worked", 2))
        training_times_last_year = st.slider("Training Times Last Year", 0, 6, a.get("training_times_last_year", 2))

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("⬅ Back")
        with col2:
            submitted = st.form_submit_button("🚀 Predict Attrition Risk")

        if back:
            a.update({
                "job_satisfaction": job_satisfaction, "env_satisfaction": env_satisfaction,
                "relationship_satisfaction": relationship_satisfaction, "work_life_balance": work_life_balance,
                "total_working_years": total_working_years, "years_at_company": years_at_company,
                "years_in_current_role": years_in_current_role, "years_since_promotion": years_since_promotion,
                "years_with_curr_manager": years_with_curr_manager, "num_companies_worked": num_companies_worked,
                "training_times_last_year": training_times_last_year,
            })
            st.session_state.step = 3
            st.rerun()

        if submitted:
            a.update({
                "job_satisfaction": job_satisfaction, "env_satisfaction": env_satisfaction,
                "relationship_satisfaction": relationship_satisfaction, "work_life_balance": work_life_balance,
                "total_working_years": total_working_years, "years_at_company": years_at_company,
                "years_in_current_role": years_in_current_role, "years_since_promotion": years_since_promotion,
                "years_with_curr_manager": years_with_curr_manager, "num_companies_worked": num_companies_worked,
                "training_times_last_year": training_times_last_year,
            })

            input_dict = {
                "Age": a["age"],
                "BusinessTravel": encoders["BusinessTravel"].transform([a["business_travel"]])[0],
                "DailyRate": a["daily_rate"],
                "Department": encoders["Department"].transform([a["department"]])[0],
                "DistanceFromHome": a["distance_from_home"],
                "Education": a["education"],
                "EducationField": encoders["EducationField"].transform([a["education_field"]])[0],
                "EmployeeNumber": a["employee_number"],
                "EnvironmentSatisfaction": a["env_satisfaction"],
                "Gender": encoders["Gender"].transform([a["gender"]])[0],
                "HourlyRate": a["hourly_rate"],
                "JobInvolvement": a["job_involvement"],
                "JobLevel": a["job_level"],
                "JobRole": encoders["JobRole"].transform([a["job_role"]])[0],
                "JobSatisfaction": a["job_satisfaction"],
                "MaritalStatus": encoders["MaritalStatus"].transform([a["marital_status"]])[0],
                "MonthlyIncome": a["monthly_income"],
                "MonthlyRate": a["monthly_rate"],
                "NumCompaniesWorked": a["num_companies_worked"],
                "OverTime": encoders["OverTime"].transform([a["overtime"]])[0],
                "PercentSalaryHike": a["percent_salary_hike"],
                "PerformanceRating": a["performance_rating"],
                "RelationshipSatisfaction": a["relationship_satisfaction"],
                "StockOptionLevel": a["stock_option_level"],
                "TotalWorkingYears": a["total_working_years"],
                "TrainingTimesLastYear": a["training_times_last_year"],
                "WorkLifeBalance": a["work_life_balance"],
                "YearsAtCompany": a["years_at_company"],
                "YearsInCurrentRole": a["years_in_current_role"],
                "YearsSinceLastPromotion": a["years_since_promotion"],
                "YearsWithCurrManager": a["years_with_curr_manager"],
            }

            input_df = pd.DataFrame([input_dict])[features]

            # Apply the same StandardScaler used during training
            input_scaled = scaler.transform(input_df.values)

            # Support both scikit-learn/XGBoost models (predict_proba) and
            # Keras/TensorFlow models (predict returns probabilities directly)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_scaled)[0][1]
            else:
                raw_output = model.predict(input_scaled)
                proba = float(np.array(raw_output).ravel()[0])

            prediction = "Likely to Leave ⚠️" if proba >= 0.5 else "Likely to Stay ✅"

            st.session_state.result = {
                "proba": proba,
                "prediction": prediction,
                "input_dict": input_dict,
            }

# ----------------------------
# Show the result (persists on screen after the form reruns)
# ----------------------------
if "result" in st.session_state and st.session_state.step == 4:
    r = st.session_state.result
    st.subheader("Prediction Result")
    st.metric("Attrition Probability", f"{r['proba']*100:.1f}%")

    if r["proba"] >= 0.5:
        st.error(r["prediction"])
    else:
        st.success(r["prediction"])

    st.progress(min(int(r["proba"] * 100), 100))

    with st.expander("See input summary"):
        st.write(r["input_dict"])