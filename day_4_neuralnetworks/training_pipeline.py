import os

def run_pipeline():
    print("=======================================")
    print("   DAY 4: DEEP LEARNING PIPELINE      ")
    print("=======================================")

    # Step 1
    print("\n[Step 1] Running Manual Neural Network...")
    os.system("python neural_network_scratch.py")

    # Step 2
    print("\n[Step 2] Training CNN on MNIST...")
    os.system("python cnn_classifier.py")

    # Step 3
    print("\n[Step 3] Applying Transfer Learning...")
    os.system("python transfer_learning.py")

    print("\n=======================================")
    print("   PIPELINE COMPLETE. CHECK FILES.     ")
    print("=======================================")

if __name__ == "__main__":
    run_pipeline()
