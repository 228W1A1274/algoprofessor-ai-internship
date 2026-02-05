import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def train_cnn():
    print("--- Loading MNIST Data for CNN ---")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and Scale
    X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    print("--- Building CNN Architecture ---")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    print("--- Training CNN ---")
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    # Save Model & Proof
    model.save("models/cnn_mnist.keras")
    
    with open("results_cnn.txt", "w") as f:
        f.write(f"CNN MNIST Accuracy: {test_acc*100:.2f}%")
    print("\n[Success] Results saved to results_cnn.txt")

if __name__ == "__main__":
    train_cnn()
