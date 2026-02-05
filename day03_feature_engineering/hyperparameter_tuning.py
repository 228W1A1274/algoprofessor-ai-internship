import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

def tune_hyperparameters():
    print("--- Starting Grid Search (Hyperparameter Tuning) ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('titanic_cleaned.csv')
    except FileNotFoundError:
        print("Error: titanic_cleaned.csv not found.")
        return

    X = df.drop(columns=['survived'])
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define the Model
    rf = RandomForestClassifier(random_state=42)

    # 3. Define the Grid (The settings to test)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # 4. Run Grid Search (Brute Force)
    # verbose=1 keeps the output clean, verbose=2 shows every step
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # 5. Get Results
    best_params = grid_search.best_params_
    best_acc = grid_search.best_score_ * 100

    print("\nBest Parameters Found:")
    print(best_params)
    print(f"Best Accuracy: {best_acc:.2f}%")

    # --- NEW: SAVE RESULTS TO FILE ---
    with open("tuning_results.txt", "w") as f:
        f.write("--- Hyperparameter Tuning Results ---\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(f"Best Parameters: {best_params}\n")
    
    print("\n[Success] Results saved to 'tuning_results.txt'")

if __name__ == "__main__":
    tune_hyperparameters()
