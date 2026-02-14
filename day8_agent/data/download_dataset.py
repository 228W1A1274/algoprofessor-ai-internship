"""
Download Public Healthcare Dataset - FINAL WORKING VERSION
Day 8 - Real Hospital Data
"""

import pandas as pd
import numpy as np
import os

def create_hospital_dataset():
    """
    Create a realistic hospital dataset with 500 patients
    """
    print("📥 Creating hospital dataset...")
    
    np.random.seed(42)
    n_patients = 500
    
    # Patient Demographics
    data = {
        'patient_id': range(1001, 1001 + n_patients),
        'name': [f"Patient_{i}" for i in range(1, n_patients + 1)],
        'age': np.random.randint(18, 85, n_patients),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        
        # Medical Conditions
        'hypertension': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'heart_disease': np.random.choice([0, 1], n_patients, p=[0.85, 0.15]),
        'diabetes': np.random.choice([0, 1], n_patients, p=[0.75, 0.25]),
        
        # Health Metrics
        'avg_glucose_level': np.random.randint(70, 200, n_patients),
        'bmi': np.round(np.random.uniform(18.5, 38.0, n_patients), 1),
        'blood_pressure_systolic': np.random.randint(90, 180, n_patients),
        'blood_pressure_diastolic': np.random.randint(60, 120, n_patients),
        
        # Lifestyle
        'smoking_status': np.random.choice(
            ['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
            n_patients,
            p=[0.45, 0.25, 0.20, 0.10]
        ),
    }
    
    # Hospital Assignment Data
    doctors = [
        'Dr. Sarah Johnson', 'Dr. Michael Chen', 'Dr. Emily Rodriguez', 
        'Dr. James Wilson', 'Dr. Lisa Anderson', 'Dr. David Martinez',
        'Dr. Jennifer Lee', 'Dr. Robert Brown', 'Dr. Amanda Taylor',
        'Dr. Christopher Moore'
    ]
    
    departments = [
        'Cardiology', 'Neurology', 'Internal Medicine', 
        'Emergency', 'General Practice', 'Orthopedics',
        'Pediatrics', 'Surgery', 'Oncology', 'Radiology'
    ]
    
    specializations = {
        'Cardiology': ['Dr. Sarah Johnson', 'Dr. Michael Chen'],
        'Neurology': ['Dr. Emily Rodriguez', 'Dr. James Wilson'],
        'Internal Medicine': ['Dr. Lisa Anderson', 'Dr. David Martinez'],
        'Emergency': ['Dr. Jennifer Lee', 'Dr. Robert Brown'],
        'General Practice': ['Dr. Amanda Taylor', 'Dr. Christopher Moore'],
        'Orthopedics': ['Dr. Sarah Johnson', 'Dr. Jennifer Lee'],
        'Pediatrics': ['Dr. Emily Rodriguez', 'Dr. Amanda Taylor'],
        'Surgery': ['Dr. Michael Chen', 'Dr. Robert Brown'],
        'Oncology': ['Dr. Lisa Anderson', 'Dr. David Martinez'],
        'Radiology': ['Dr. James Wilson', 'Dr. Christopher Moore']
    }
    
    data['department'] = np.random.choice(departments, n_patients)
    
    # Assign doctors based on department
    data['doctor_assigned'] = [
        np.random.choice(specializations[dept])
        for dept in data['department']
    ]
    
    # Medical Procedures & Costs
    procedures = {
        'Cardiology': ['ECG Test', 'Echocardiogram', 'Stress Test', 'Angiography'],
        'Neurology': ['MRI Scan', 'CT Scan', 'EEG', 'Nerve Conduction Study'],
        'Internal Medicine': ['Blood Test', 'Urinalysis', 'Physical Exam', 'Vaccination'],
        'Emergency': ['X-Ray', 'Emergency Care', 'IV Fluids', 'Wound Care'],
        'General Practice': ['Consultation', 'Blood Test', 'Physical Exam', 'Vaccination'],
        'Orthopedics': ['X-Ray', 'MRI Scan', 'Cast Application', 'Physical Therapy'],
        'Pediatrics': ['Vaccination', 'Growth Assessment', 'Blood Test', 'Physical Exam'],
        'Surgery': ['Appendectomy', 'Hernia Repair', 'Gallbladder Removal', 'Orthopedic Surgery'],
        'Oncology': ['Chemotherapy', 'Radiation', 'Biopsy', 'CT Scan'],
        'Radiology': ['X-Ray', 'MRI Scan', 'CT Scan', 'Ultrasound']
    }
    
    procedure_costs = {
        'ECG Test': 50, 'Echocardiogram': 300, 'Stress Test': 200, 'Angiography': 1500,
        'MRI Scan': 800, 'CT Scan': 600, 'EEG': 250, 'Nerve Conduction Study': 350,
        'Blood Test': 30, 'Urinalysis': 20, 'Physical Exam': 100, 'Vaccination': 40,
        'X-Ray': 120, 'Emergency Care': 500, 'IV Fluids': 150, 'Wound Care': 200,
        'Consultation': 150, 'Cast Application': 180, 'Physical Therapy': 100,
        'Appendectomy': 3500, 'Hernia Repair': 4000, 'Gallbladder Removal': 5000,
        'Orthopedic Surgery': 6000, 'Chemotherapy': 2500, 'Radiation': 3000,
        'Biopsy': 400, 'Ultrasound': 250, 'Growth Assessment': 80
    }
    
    # Assign procedures based on department
    data['procedure'] = [
        np.random.choice(procedures[dept])
        for dept in data['department']
    ]
    
    data['procedure_cost'] = [
        procedure_costs[proc]
        for proc in data['procedure']
    ]
    
    # Consultation fees
    doctor_fees = {
        'Dr. Sarah Johnson': 200, 'Dr. Michael Chen': 250, 'Dr. Emily Rodriguez': 180,
        'Dr. James Wilson': 220, 'Dr. Lisa Anderson': 200, 'Dr. David Martinez': 190,
        'Dr. Jennifer Lee': 180, 'Dr. Robert Brown': 230, 'Dr. Amanda Taylor': 170,
        'Dr. Christopher Moore': 210
    }
    
    data['consultation_fee'] = [
        doctor_fees[doc]
        for doc in data['doctor_assigned']
    ]
    
    # Total cost
    data['total_cost'] = [
        proc_cost + consult_fee + np.random.randint(0, 500)
        for proc_cost, consult_fee in zip(data['procedure_cost'], data['consultation_fee'])
    ]
    
    # Diagnoses
    diagnoses = [
        'Hypertension', 'Type 2 Diabetes', 'Coronary Artery Disease', 'Asthma',
        'Arthritis', 'Migraine', 'Pneumonia', 'Fracture', 'Healthy Checkup',
        'Influenza', 'COVID-19', 'Back Pain', 'Obesity', 'Anxiety', 'Depression'
    ]
    data['diagnosis'] = np.random.choice(diagnoses, n_patients)
    
    # Room assignments
    data['room_number'] = [
        f"{np.random.randint(1, 6)}{np.random.randint(10, 50):02d}"
        for _ in range(n_patients)
    ]
    
    # Admission status
    data['admission_status'] = np.random.choice(
        ['Admitted', 'Outpatient', 'Discharged'],
        n_patients,
        p=[0.3, 0.5, 0.2]
    )
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/healthcare_data.csv', index=False)
    
    print(f"✅ Hospital dataset created successfully!")
    print(f"\n📊 Dataset Summary:")
    print(f"   • Total Patients: {len(df)}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"   • Departments: {', '.join(departments[:5])}...")
    print(f"   • Doctors: {len(doctors)}")
    print(f"\n👥 Sample Patient Record:")
    print(df[['patient_id', 'name', 'age', 'department', 'doctor_assigned', 'diagnosis', 'total_cost']].head(1).to_string(index=False))
    
    return df


def main():
    """Create dataset"""
    print("\n" + "="*60)
    print("HOSPITAL DATASET CREATION")
    print("="*60 + "\n")
    
    df = create_hospital_dataset()
    
    print("\n" + "="*60)
    print("✅ Dataset ready at: data/healthcare_data.csv")
    print("="*60)


if __name__ == "__main__":
    main()