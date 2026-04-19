import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ================= LOAD DATA =================
df = pd.read_csv("data/employee_data.csv")

X = df.drop(["performance", "employee_id"], axis=1)
y = df["performance"]

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= PREPROCESSING =================
cat_cols = ["department"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

# ================= MODEL =================
model = RandomForestClassifier(n_estimators=300, random_state=42)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

# ================= TRAIN =================
pipeline.fit(X_train, y_train)

# ================= PREDICT =================
y_pred = pipeline.predict(X_test)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ================= REPORT =================
print(classification_report(y_test, y_pred))

# ================= SAVE MODEL =================
joblib.dump(pipeline, "models/employee_model.pkl")

print("Model saved successfully!")