import numpy as np
import pandas as pd

np.random.seed(42)

n = 1200

df = pd.DataFrame({
    "employee_id": range(1, n+1),
    "age": np.random.randint(22, 60, n),
    "experience_years": np.random.randint(0, 35, n),
    "department": np.random.choice(["HR", "IT", "Finance", "Sales"], n),
    "salary": np.random.randint(30000, 150000, n),
    "training_hours": np.random.randint(0, 100, n),
    "projects_completed": np.random.randint(0, 20, n),
    "attendance_rate": np.round(np.random.uniform(0.5, 1.0, n), 2),
    "last_rating": np.random.randint(1, 6, n)
})

# 🎯 Performance logic simulation
def performance(row):
    score = (
        row["projects_completed"] * 2 +
        row["attendance_rate"] * 10 +
        row["training_hours"] * 0.05 +
        row["last_rating"] * 3
    )

    if score > 25:
        return "High"
    elif score > 15:
        return "Medium"
    else:
        return "Low"

df["performance"] = df.apply(performance, axis=1)

df.to_csv("employee_data.csv", index=False)

print("✅ Dataset generated successfully!")