import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # He Initialization (Crucial for high accuracy)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, learning_rate=0.1):
        m = y_true.shape[0]
        dZ2 = self.A2 - y_true
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, lr=0.1):
        print(f"--- Training NN from Scratch on Digits Data ---")
        final_acc = 0
        for i in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate=lr)
            
            if i % 100 == 0:
                predictions = np.argmax(y_pred, axis=1)
                true_labels = np.argmax(y, axis=1)
                final_acc = np.mean(predictions == true_labels)
                print(f"Epoch {i}: Accuracy = {final_acc*100:.2f}%")
        return final_acc

if __name__ == "__main__":
    # 1. Load Real Data
    digits = load_digits()
    X = digits.data
    y = digits.target.reshape(-1, 1)

    # 2. Scale Data (StandardScaler is vital for convergence)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)

    # 3. Train
    nn = SimpleNN(input_size=64, hidden_size=64, output_size=10)
    final_acc = nn.train(X, y_onehot, epochs=2000, lr=0.5)
    
    # 4. Save Proof
    with open("results_scratch.txt", "w") as f:
        f.write(f"Manual NN Accuracy: {final_acc*100:.2f}%")
    print(f"\n[Success] Results saved to results_scratch.txt")
