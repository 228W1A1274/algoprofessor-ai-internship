import pandas as pd
import os


class HospitalTools:

    def __init__(self, data_path="data/healthcare_data.csv"):
        self.data_path = data_path
        self.df = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            print("⚠ Dataset not found. Run download_dataset.py first.")
            return pd.DataFrame()
        return pd.read_csv(self.data_path)

    # =====================================================
    # TOOL 1: SMART DATABASE SEARCH
    # =====================================================
    def rag_search_tool(self, query: str) -> str:
        if self.df.empty:
            return "No data loaded."

        query = query.lower()

        # ---- 1. Doctor consultation fee lookup ----
        for doctor in self.df['doctor_assigned'].unique():
            if doctor.lower() in query:
                fee = self.df[self.df['doctor_assigned'] == doctor]['consultation_fee'].iloc[0]
                return f"{doctor} consultation fee is ${fee}"

        # ---- 2. Department / Specialization lookup ----
        for dept in self.df['department'].unique():
            if dept.lower() in query:
                doctors = self.df[self.df['department'] == dept]['doctor_assigned'].unique()
                return f"Doctors in {dept} department:\n" + "\n".join(doctors)

        # ---- 3. Procedure cost lookup ----
        for procedure in self.df['procedure'].unique():
            if procedure.lower() in query:
                cost = self.df[self.df['procedure'] == procedure]['procedure_cost'].iloc[0]
                return f"{procedure} costs ${cost}"

        # ---- 4. Age statistics ----
        if "age" in query:
            return f"Average patient age is {self.df['age'].mean():.1f} years"

        # ---- 5. Hypertension count ----
        if "hypertension" in query:
            count = self.df['hypertension'].sum()
            return f"Total hypertension cases: {count}"

        # ---- 6. Total cost statistics ----
        if "total cost" in query or "average cost" in query:
            return f"Average total cost is ${self.df['total_cost'].mean():.2f}"

        return "No matching data found in hospital database."

    # =====================================================
    # TOOL 2: CALCULATOR
    # =====================================================
    def calculator_tool(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": {}})
            return f"Result: {result}"
        except Exception:
            return "Invalid calculation"

    # =====================================================
    # TOOL 3: DATA ANALYSIS
    # =====================================================
    def data_analysis_tool(self, query: str) -> str:
        if self.df.empty:
            return "No data loaded."

        return self.df.describe().to_string()

    # =====================================================
    # TOOL 4: SEARCH PATIENT
    # =====================================================
    def search_patient_tool(self, patient_name: str) -> str:
        if self.df.empty:
            return "No data loaded."

        patient_name = patient_name.lower()
        matches = self.df[self.df['name'].str.lower().str.contains(patient_name)]

        if matches.empty:
            return "Patient not found."

        row = matches.iloc[0]

        return (
            f"Patient ID: {row['patient_id']}\n"
            f"Name: {row['name']}\n"
            f"Age: {row['age']}\n"
            f"Gender: {row['gender']}\n"
            f"Department: {row['department']}\n"
            f"Doctor Assigned: {row['doctor_assigned']}\n"
            f"Diagnosis: {row['diagnosis']}\n"
            f"Room Number: {row['room_number']}\n"
            f"Total Cost: ${row['total_cost']}"
        )

    # =====================================================
    # TOOL 5: DOCTOR AVAILABILITY
    # =====================================================
    def check_doctor_availability_tool(self, specialization: str) -> str:
        if self.df.empty:
            return "No data loaded."

        specialization = specialization.lower()

        if specialization:
            matches = self.df[self.df['department'].str.lower().str.contains(specialization)]
        else:
            matches = self.df

        if matches.empty:
            return "No doctors found for this specialization."

        doctors = matches['doctor_assigned'].unique()

        return "Available Doctors:\n" + "\n".join(doctors)


# =====================================================
# GLOBAL INSTANCE
# =====================================================
hospital_tools = HospitalTools()


# =====================================================
# FUNCTION WRAPPERS (FOR LANGCHAIN)
# =====================================================
def rag_search(query: str):
    return hospital_tools.rag_search_tool(query)


def calculator(expression: str):
    return hospital_tools.calculator_tool(expression)


def analyze_data(query: str):
    return hospital_tools.data_analysis_tool(query)


def search_patient(name: str):
    return hospital_tools.search_patient_tool(name)


def check_doctor_availability(specialization: str):
    return hospital_tools.check_doctor_availability_tool(specialization)
