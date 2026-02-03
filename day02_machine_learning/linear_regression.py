import numpy as np
import matplotlib.pyplot as plt
class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iterations=1000):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.weights=None
        self.bias=0
        self.loss_history=[]

    def fit(self,x,y):
            n_samples,n_features=X.shape
            self.weights=np.zeros(n_features)
            self.bias=0
            for i in range(self.n_iterations):
                y_predicted=np.dot(X,self.weights)+self.bias
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) # Gradient for weight
                db = (1 / n_samples) * np.sum(y_predicted - y)
                self.weights-=self.learning_rate*dw
                self.bias-=self.learning_rate*db
                loss=(1/n_samples)*np.sum((y_predicted - y)**2)
                self.loss_history.append(loss)
                if i % 100 == 0:
                    print(f"Iteration {i}: Loss {loss:.4f}")
    def predict(self,X):
            return np.dot(X, self.weights) + self.bias
if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]]) 
    #y includes some noise for testing and traing teh model
    y = np.array([12, 22, 35, 38, 52])
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    print("\n--- Predictions ---")
    user_input = np.array([[6]]) # Predicting score for 6 hours of study
    prediction = model.predict(user_input)
    print(f"If you study for 6 hours, predicted score: {prediction[0]:.2f}")
    plt.scatter(X, y, color='blue', label='Actual Data')
    plt.plot(X, model.predict(X), color='red', label='Prediction Line')

    plt.xlabel('Hours')
    plt.ylabel('Score')
    plt.legend()
    plt.show()