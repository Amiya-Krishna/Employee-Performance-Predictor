import joblib
import pandas as pd

model = joblib.load("models/employee_model.pkl")

sample = pd.DataFrame([{
    "age": 30,
    "experience_years": 5,
    "department": "IT",
    "salary": 70000,
    "training_hours": 40,
    "projects_completed": 10,
    "attendance_rate": 0.9,
    "last_rating": 4
}])

print("Prediction:", model.predict(sample)[0])