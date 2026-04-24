# Fashion MNIST Image Classifier using Keras (CNN)
# This notebook builds, trains, and evaluates a CNN on the Fashion MNIST dataset

# =========================
# 1. Imports
# =========================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow version:", tf.__version__)

# =========================
# 2. Load Dataset
# =========================
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Training data shape:", x_train.shape)

# =========================
# 3. Preprocessing
# =========================
# Normalize pixel values (0-255 -> 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add channel dimension (28x28 -> 28x28x1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

print("New shape:", x_train.shape)

# =========================
# 4. Visualize Data
# =========================
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[i].squeeze(), cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()

# =========================
# 5. Build CNN Model
# =========================
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# =========================
# 6. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 7. Train Model
# =========================
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# =========================
# 8. Evaluate Model
# =========================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy:", test_acc)

# =========================
# 9. Plot Training History
# =========================
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

# =========================
# 10. Make Predictions
# =========================
predictions = model.predict(x_test)

# Display predictions for first 5 images
for i in range(5):
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}\nActual: {class_names[y_test[i]]}")
    plt.axis('off')
    plt.show()

# =========================
# 11. Save Model
# =========================
model.save('fashion_mnist_cnn.h5')
print("Model saved as fashion_mnist_cnn.h5")
