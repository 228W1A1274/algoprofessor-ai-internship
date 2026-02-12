
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Import the 3 Brains
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def compare_real_models():
    print("--- 1. Loading Real Customer Data ---")
    file_path = 'day02_machine_learning/social_network_ads.csv'
    
    if not os.path.exists(file_path):
        print("Error: CSV file not found!")
        return

    df = pd.read_csv(file_path)
    print("Data Loaded Successfully!")
    print(df.head())

    # Inputs: Age and Salary
    X = df[['Age', 'EstimatedSalary']]
    # Output: Purchased (0 or 1)
    y = df['Purchased']

    # Split into Training (Study) and Testing (Exam)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # CRITICAL STEP: Feature Scaling
    # Salaries (140,000) are huge compared to Age (40). 
    # This confuses the SVM model. We must "scale" them to be similar sizes.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- 2. Training Models ---")
    
    # Model A: Decision Tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train_scaled, y_train)
    tree_acc = accuracy_score(y_test, tree.predict(X_test_scaled))
    print(f"Decision Tree Accuracy: {tree_acc*100:.1f}%")

    # Model B: Random Forest (The Committee)
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train_scaled, y_train)
    forest_acc = accuracy_score(y_test, forest.predict(X_test_scaled))
    print(f"Random Forest Accuracy: {forest_acc*100:.1f}%")

    # Model C: SVM (The Separator)
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    print(f"SVM Accuracy:           {svm_acc*100:.1f}%")

    print("\n--- 3. Visualizing the Comparison ---")
    output_dir = 'day02_machine_learning/outputs'
    os.makedirs(output_dir, exist_ok=True)

    models = ['Decision Tree', 'Random Forest', 'SVM']
    scores = [tree_acc, forest_acc, svm_acc]

    plt.figure(figsize=(8, 5))
    # Draw the bar chart
    ax = sns.barplot(x=models, y=scores, palette='viridis')
    plt.ylim(0, 1.1) # Scale from 0% to 110% just to make it look clean
    plt.title('Real-World Model Comparison: Who Predicts Buyers Best?')
    plt.ylabel('Accuracy Score')

    # Add the text numbers on top of the bars
    for i, score in enumerate(scores):
        ax.text(i, score + 0.02, f'{score*100:.1f}%', ha='center', fontweight='bold')

    save_path = f'{output_dir}/real_model_comparison.png'
    plt.savefig(save_path)
    print(f"Comparison Chart saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    compare_real_models()