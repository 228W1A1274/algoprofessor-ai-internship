import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression # The Professional Tool
from sklearn.model_selection import train_test_split

def predict_salary():
    print("--- 1. Loading the Dataset ---")
    # Instead of creating fake numbers, we read the CSV file
    # This is how 99% of real AI work starts
    file_path = 'day02_machine_learning/salary_data.csv'
    
    if not os.path.exists(file_path):
        print("Error: CSV file not found! Did you create salary_data.csv?")
        return

    df = pd.read_csv(file_path)
    print("Data Loaded Successfully!")
    print(df.head()) # Show first 5 rows

    # Separate Input (X) and Output (y)
    # X = Years of Experience (Independent Variable)
    # y = Salary (Dependent Variable)
    X = df[['YearsExperience']] 
    y = df['Salary']

    print("\n--- 2. Training the Linear Regression Model ---")
    # We use scikit-learn, which has the math pre-written for us
    model = LinearRegression()
    
    # .fit() is where the AI learns the pattern (y = mx + b)
    model.fit(X, y)
    
    print("Model Trained!")
    print(f"Coefficient (Slope): {model.coef_[0]:.2f} (For every 1 year exp, salary increases by this amount)")
    print(f"Intercept (Bias): {model.intercept_:.2f} (Base salary for 0 years exp)")

    print("\n--- 3. Testing (Prediction) ---")
    # Let's predict the salary for someone with 12 and 15 years experience
    test_years = pd.DataFrame({'YearsExperience': [12, 15]})
    predictions = model.predict(test_years)
    
    print(f"Predicted Salary for 12 Years: ${predictions[0]:,.2f}")
    print(f"Predicted Salary for 15 Years: ${predictions[1]:,.2f}")

    print("\n--- 4. Visualizing the Result ---")
    # Setup Output Folder
    output_dir = 'day02_machine_learning/outputs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    # Plot the Real Data (Blue Dots)
    plt.scatter(X, y, color='blue', label='Real Employee Data')
    
    # Plot the AI's Prediction Line (Red Line)
    plt.plot(X, model.predict(X), color='red', label='Regression Line (Prediction)')
    
    plt.title('Salary vs Experience (Linear Regression)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary ($)')
    plt.legend()
    plt.grid(True)
    
    # Save Graph
    save_path = f'{output_dir}/salary_prediction_real.png'
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    predict_salary()