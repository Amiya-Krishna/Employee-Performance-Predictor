from fpdf import FPDF
import os
import datetime

class HRReport:
    def __init__(self, output_path="outputs/hr_report.pdf"):
        self.output_path = output_path

    def generate(self, employee_data, prediction, accuracy):
        pdf = FPDF()
        pdf.add_page()

        # Title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "HR ANALYTICS REPORT", ln=True, align="C")

        pdf.ln(10)

        # Date
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Date: {datetime.datetime.now()}", ln=True)

        pdf.ln(5)

        # Employee Info
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Employee Details:", ln=True)

        pdf.set_font("Arial", size=11)
        for key, value in employee_data.items():
            pdf.cell(200, 8, f"{key}: {value}", ln=True)

        pdf.ln(5)

        # Prediction
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, "Prediction Result:", ln=True)

        pdf.set_font("Arial", size=11)
        pdf.cell(200, 8, f"Predicted Performance: {prediction}", ln=True)

        pdf.ln(5)

        # Model accuracy
        pdf.cell(200, 8, f"Model Accuracy: {accuracy}", ln=True)

        pdf.ln(10)

        # Images (charts)
        if os.path.exists("outputs/confusion_matrix.png"):
            pdf.image("outputs/confusion_matrix.png", w=180)

        pdf.ln(10)

        if os.path.exists("outputs/feature_importance.png"):
            pdf.image("outputs/feature_importance.png", w=180)

        # Save PDF
        pdf.output(self.output_path)

        return self.output_path