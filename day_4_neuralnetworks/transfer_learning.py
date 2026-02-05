import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, Input
import numpy as np
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

def run_transfer_learning():
    print("--- 1. Loading FULL CIFAR-10 Data ---")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Cast to float
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    print(f"Training on {len(X_train)} images...")

    print("--- 2. Building Architecture with Augmentation ---")
    
    inputs = Input(shape=(32, 32, 3))
    
    # Data Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    # Resize to 96x96
    x = layers.Lambda(lambda img: tf.image.resize(img, (96, 96)))(x)
    
    # Preprocess
    x = layers.Lambda(lambda img: tf.keras.applications.mobilenet_v2.preprocess_input(img))(x)

    # Base Model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False 

    # Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("--- 3. Phase 1: Training Head ---")
    history = model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

    print("--- 4. Phase 2: Fine-Tuning ---")
    base_model.trainable = True
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    history_fine = model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    # --- THIS WAS THE BROKEN LINE ---
    model.save("models/transfer_learning_mobilenet.keras")
    
    with open("results_transfer.txt", "w") as f:
        f.write(f"Transfer Learning Accuracy: {test_acc*100:.2f}%")
    print("\n[Success] Results saved to results_transfer.txt")

if __name__ == "__main__":
    run_transfer_learning()
