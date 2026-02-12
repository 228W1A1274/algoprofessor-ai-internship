from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Day 3 ML Performance Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="1. Feature Engineering: Completed (Imputation, Scaling)", ln=True)
    pdf.cell(200, 10, txt="2. Models Trained: XGBoost, LightGBM, Random Forest", ln=True)
    pdf.cell(200, 10, txt="3. Best Accuracy Achieved: ~82% (XGBoost)", ln=True)

    pdf.cell(200, 10, txt="4. Visualizations (PCA & t-SNE):", ln=True)
    # Ensure the image exists before adding
    try:
        pdf.image("dim_reduction_plot.png", x=10, y=60, w=190)
    except:
        pdf.cell(200, 10, txt="[Plot not found - Run dimensionality_reduction.py first]", ln=True)

    pdf.output("performance_report.pdf")
    print("PDF Report generated successfully.")

if __name__ == "__main__":
    create_pdf()
