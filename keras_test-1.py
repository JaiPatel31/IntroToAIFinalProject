import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Data
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# Normalize to [0,1]
train_images = train_images.astype("float32") / 255.0
test_images  = test_images.astype("float32") / 255.0

# Train/validation split
val_images = train_images[-10000:]
val_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]

# Model
'''model = keras.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])'''



 # CNN Model
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),               # helps val accuracy generalize
    layers.Dense(10, activation="softmax"),
])



learning_rate = 1e-3

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Training with history
history = model.fit(
    train_images, train_labels,
    epochs=15,
    batch_size=64,
    validation_data=(val_images, val_labels),
)

# Plot Curves
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

plt.figure()
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.show()

# Test evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)

# Predict on test set
y_prob = model.predict(test_images)
y_pred = np.argmax(y_prob, axis=1)

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(6,6))
disp.plot(values_format='d', cmap="Blues", colorbar=False)
plt.title("Confusion Matrix – Final CNN")
plt.show()

# Misclassified samples
mis_idx = np.where(y_pred != test_labels)[0]

plt.figure(figsize=(8,8))
for i, idx in enumerate(mis_idx[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test_images[idx].squeeze(), cmap="gray")
    plt.title(f"True: {test_labels[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()