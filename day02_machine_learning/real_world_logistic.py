import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def classify_flowers():
    print("--- 1. Loading the Iris Dataset ---")
    file_path = 'day02_machine_learning/iris_subset.csv'
    
    if not os.path.exists(file_path):
        print("Error: CSV file not found!")
        return

    df = pd.read_csv(file_path)
    print("Data Loaded!")
    print(df.head())

    # Inputs (Sepal Length, Sepal Width)
    X = df[['SepalLength', 'SepalWidth']]
    # Output (Species: 0 or 1)
    y = df['Species']

    print("\n--- 2. Training Logistic Regression ---")
    model = LogisticRegression()
    model.fit(X, y)
    
    print("Model Trained!")
    
    print("\n--- 3. Testing Predictions ---")
    # Let's find a random flower and see if the AI knows what it is
    # Flower A: Length 5.1, Width 3.5 (Should be 0/Setosa)
    # Flower B: Length 6.0, Width 3.0 (Should be 1/Versicolor)
    test_flowers = pd.DataFrame([[5.1, 3.5], [6.0, 3.0]], columns=['SepalLength', 'SepalWidth'])
    
    predictions = model.predict(test_flowers)
    print(f"Flower A Prediction: {predictions[0]} (Expected 0)")
    print(f"Flower B Prediction: {predictions[1]} (Expected 1)")

    print("\n--- 4. Visualizing the Decision Boundary ---")
    output_dir = 'day02_machine_learning/outputs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    # 1. Plot the actual flowers
    sns.scatterplot(x='SepalLength', y='SepalWidth', hue='Species', data=df, s=100, palette=['blue', 'red'])
    
    # 2. Draw the Decision Boundary (The line separating Blue vs Red)
    # This math calculates the slope and intercept of the separation line
    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2
    
    # Plot the line
    x_vals = np.array([4.0, 7.5])
    y_vals = m * x_vals + c
    plt.plot(x_vals, y_vals, 'k--', lw=2, label="Decision Boundary")
    
    plt.title('Logistic Regression: Classifying Flowers')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend(title='Species (0=Blue, 1=Red)')
    
    save_path = f'{output_dir}/iris_classification_real.png'
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    classify_flowers()