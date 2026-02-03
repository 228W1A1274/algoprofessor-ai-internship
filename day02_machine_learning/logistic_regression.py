import numpy as np
import matplotlib.pyplot as plt
import os  # Imported to handle folder creation

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            loss = -np.mean(y * np.log(y_predicted + 1e-9) + (1-y) * np.log(1-y_predicted + 1e-9))
            self.loss_history.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        class_predictions = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(class_predictions)

if __name__ == "__main__":
    print("--- Training Logistic Regression (Binary Classification) ---")
    
    # 1. Fake Data
    X = np.array([[1], [1.5], [2], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # 2. Train Model
    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=2000)
    model.fit(X, y)

    # 3. Test Prediction
    user_input = np.array([[3]])
    prediction = model.predict(user_input)
    print(f"Prediction for 3 hours: {prediction[0]} (0=Fail, 1=Pass)")

    # 4. Visualize AND Save
    # Create the output directory if it doesn't exist
    output_dir = 'day02_machine_learning/outputs'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6)) # Make the chart a bit bigger
    plt.scatter(X, y, color='blue', label='Actual Data (0 or 1)')
    
    # Generate smooth curve
    X_test = np.linspace(0, 7, 100).reshape(-1, 1)
    linear_part = np.dot(X_test, model.weights) + model.bias
    y_prob = model._sigmoid(linear_part)
    
    plt.plot(X_test, y_prob, color='red', label='Sigmoid Probability Curve')
    plt.axhline(y=0.5, color='green', linestyle='--', label='Decision Boundary (0.5)')
    
    plt.title('Logistic Regression: Pass vs Fail')
    plt.xlabel('Hours Studied')
    plt.ylabel('Probability of Passing')
    plt.legend()
    
    # SAVE THE FILE
    save_path = f'{output_dir}/logistic_regression_curve.png'
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    
    # Show it (optional, close the window to finish the script)
    plt.show()