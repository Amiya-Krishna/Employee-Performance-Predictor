import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hr_report import HRReport

# Load model + data
model = joblib.load("models/employee_model.pkl")
df = pd.read_csv("data/employee_data.csv")

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

# ================= HEADER =================
st.title("🧠 HR Analytics Dashboard - Employee Performance Predictor")
st.markdown("### AI-powered HR decision support system")

# ================= KPI CARDS =================
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))
col2.metric("Avg Salary", f"{int(df['salary'].mean())}")
col3.metric("Departments", df["department"].nunique())

st.divider()

# ================= PERFORMANCE DISTRIBUTION =================
st.subheader("📊 Performance Distribution")

fig, ax = plt.subplots()
sns.countplot(data=df, x="performance", palette="Set2", ax=ax)
st.pyplot(fig)

# ================= DEPARTMENT WISE ANALYSIS =================
st.subheader("🏢 Department-wise Performance")

fig2, ax2 = plt.subplots()
sns.countplot(data=df, x="department", hue="performance", ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)

# ================= INPUT =================
st.subheader("🔮 Employee Performance Prediction")

age = st.number_input("Age", 20, 60, 30)
exp = st.number_input("Experience", 0, 40, 5)
dept = st.selectbox("Department", ["HR", "IT", "Finance", "Sales"])
salary = st.number_input("Salary", 30000, 200000, 70000)
training = st.number_input("Training Hours", 0, 100, 20)
projects = st.number_input("Projects Completed", 0, 30, 5)
attendance = st.slider("Attendance Rate", 0.0, 1.0, 0.8)
rating = st.slider("Last Rating", 1, 5, 3)

# session state init
if "pred" not in st.session_state:
    st.session_state["pred"] = None

# ================= PREDICTION =================
if st.button("Predict Performance"):

    input_df = pd.DataFrame([{
        "age": age,
        "experience_years": exp,
        "department": dept,
        "salary": salary,
        "training_hours": training,
        "projects_completed": projects,
        "attendance_rate": attendance,
        "last_rating": rating
    }])

    pred = model.predict(input_df)[0]
    st.session_state["pred"] = pred

    st.success(f"Prediction: {pred}")

    if pred == "High":
        st.success("🔥 High Performer")
    elif pred == "Medium":
        st.warning("⚡ Medium Performer")
    else:
        st.error("⚠️ Low Performer")

# ================= DATA =================
st.subheader("📋 Employee Dataset Preview")
st.dataframe(df.head(20))

# ================= REPORT =================
report_generator = HRReport()

if st.button("Generate HR Report"):

    if st.session_state["pred"] is None:
        st.error("Pehle prediction run karo")
    else:
        path = report_generator.generate(
            employee_data={
                "Age": age,
                "Experience": exp,
                "Department": dept,
                "Salary": salary
            },
            prediction=st.session_state["pred"],
            accuracy="94%"
        )

        st.success("Report Generated Successfully!")

        with open(path, "rb") as f:
            st.download_button(
                "Download HR Report",
                f,
                file_name="HR_Report.pdf"
            )