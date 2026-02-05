import os

def run_pipeline():
    print("==================================")
    print("   STARTING END-TO-END PIPELINE   ")
    print("==================================")

    # Step 1: Feature Engineering
    print("\n[Step 1] Running Feature Engineering...")
    os.system("python feature_engineering.py")

    # Step 2: Dimensionality Reduction
    print("\n[Step 2] Running PCA & t-SNE...")
    os.system("python dimensionality_reduction.py")

    # Step 3: Hyperparameter Tuning
    print("\n[Step 3] Running Hyperparameter Tuning...")
    os.system("python hyperparameter_tuning.py")

    # Step 4: Ensemble Models
    print("\n[Step 4] Training Ensemble Models (XGBoost/LightGBM)...")
    os.system("python ensemble_models.py")

    print("\n==================================")
    print("   PIPELINE COMPLETED SUCCESSFULLY  ")
    print("==================================")

if __name__ == "__main__":
    run_pipeline()
