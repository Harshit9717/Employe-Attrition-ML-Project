import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Attrition Predictor", layout="wide")

st.title("💼 Employee Attrition Prediction")

gender_map = {"Male": 1, "Female": 0}
overtime_map = {"Yes": 1, "No": 0}

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 60)
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    distance = st.number_input("Distance From Home", 1, 50)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    environment_satisfaction = st.slider("Environment Satisfaction", 1, 4)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_involvement = st.slider("Job Involvement", 1, 4)
    job_level = st.slider("Job Level", 1, 5)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative",
        "Manager", "Sales Representative", "Research Director", "Human Resources"
    ])
    job_satisfaction = st.slider("Job Satisfaction", 1, 4)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    monthly_income = st.number_input("Monthly Income", 1000, 20000)

with col3:
    num_companies = st.number_input("Num Companies Worked", 0, 10)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    salary_hike = st.slider("Percent Salary Hike", 10, 25)
    performance = st.slider("Performance Rating", 1, 4)
    relationship = st.slider("Relationship Satisfaction", 1, 4)
    total_years = st.number_input("Total Working Years", 0, 40)
    work_life = st.slider("Work Life Balance", 1, 4)
    years_company = st.number_input("Years At Company", 0, 40)
    years_role = st.number_input("Years In Current Role", 0, 20)
    years_promo = st.number_input("Years Since Last Promotion", 0, 15)
    years_manager = st.number_input("Years With Current Manager", 0, 20)

if st.button("Predict"):

    data = {
        "Age": age,
        "BusinessTravel": business_travel,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender_map[gender],   # ✅ mapped
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime_map[overtime],  # ✅ mapped
        "PercentSalaryHike": salary_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": relationship,
        "TotalWorkingYears": total_years,
        "WorkLifeBalance": work_life,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_manager
    }

    try:
        response = requests.post(API_URL, json=data)
        result = response.json()

        if "prediction" in result:
            prob = result["probability"]
            pred = result["prediction"]

            st.subheader("Result")

            if pred == 1:
                st.error(f"⚠️ Employee likely to leave (Probability: {prob:.2f})")
            else:
                st.success(f"✅ Employee likely to stay (Probability: {prob:.2f})")

            st.progress(prob)

        else:
            st.warning(result)

    except Exception as e:
        st.error(f"Error: {e}")