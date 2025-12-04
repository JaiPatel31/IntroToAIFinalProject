import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

st.set_page_config(page_title="Fashion-MNIST Trainer", layout="wide")

# -------------------------
# Load & preprocess data
# -------------------------
@st.cache_data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images  = test_images.astype("float32") / 255.0

    # train/val split
    val_images = train_images[-10000:]
    val_labels = train_labels[-10000:]
    train_images = train_images[:-10000]
    train_labels = train_labels[:-10000]

    # CNN versions (extra channel)
    train_images_cnn = np.expand_dims(train_images, -1)
    val_images_cnn   = np.expand_dims(val_images, -1)
    test_images_cnn  = np.expand_dims(test_images, -1)

    return (train_images, train_labels,
            val_images, val_labels,
            test_images, test_labels,
            train_images_cnn, val_images_cnn, test_images_cnn)

(
    train_images, train_labels,
    val_images, val_labels,
    test_images, test_labels,
    train_images_cnn, val_images_cnn, test_images_cnn
) = load_data()

# -------------------------
# Model builders
# -------------------------
def build_mlp_small():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model

def build_mlp_large():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model

def build_cnn_final():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    return model

MODEL_BUILDERS = {
    "MLP (small)": build_mlp_small,
    "MLP (large)": build_mlp_large,
    "CNN (final)": build_cnn_final,
}

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Training Controls")

model_name = st.sidebar.selectbox("Model", list(MODEL_BUILDERS.keys()), index=2)
lr = st.sidebar.number_input("Learning rate", value=1e-3, format="%.5f")
batch_size = st.sidebar.number_input("Batch size", value=64, min_value=8, step=8)
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=10)
run_button = st.sidebar.button("Start training")

st.title("Fashion-MNIST Training Dashboard")
st.write("Select a model and hyperparameters in the sidebar, then click **Start training**.")

# -------------------------
# Main action
# -------------------------
if run_button:
    st.write(f"### Training: {model_name}")
    builder = MODEL_BUILDERS[model_name]
    model = builder()

    if "CNN" in model_name:
        x_train, x_val, x_test = train_images_cnn, val_images_cnn, test_images_cnn
    else:
        x_train, x_val, x_test = train_images, val_images, test_images

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Placeholders for live plots + status text
    status_placeholder = st.empty()
    col1, col2 = st.columns(2)
    loss_plot_ph = col1.empty()
    acc_plot_ph = col2.empty()

    # Storage for history
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    total_epochs = int(epochs)
    batch_size_int = int(batch_size)

    for epoch_idx in range(total_epochs):
        status_placeholder.write(f"Epoch {epoch_idx+1}/{total_epochs} ...")

        # Train ONE epoch
        h = model.fit(
            x_train, train_labels,
            validation_data=(x_val, val_labels),
            epochs=1,
            batch_size=batch_size_int,
            verbose=0,
        )

        # Append metrics
        train_loss.append(h.history["loss"][0])
        val_loss.append(h.history["val_loss"][0])
        train_acc.append(h.history["accuracy"][0])
        val_acc.append(h.history["val_accuracy"][0])

        epochs_range = range(1, len(train_loss) + 1)

        # --- Update loss plot ---
        fig1, ax1 = plt.subplots()
        ax1.plot(epochs_range, train_loss, label="train")
        ax1.plot(epochs_range, val_loss, label="val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        loss_plot_ph.pyplot(fig1)
        plt.close(fig1)

        # --- Update accuracy plot ---
        fig2, ax2 = plt.subplots()
        ax2.plot(epochs_range, train_acc, label="train")
        ax2.plot(epochs_range, val_acc, label="val")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        acc_plot_ph.pyplot(fig2)
        plt.close(fig2)

    status_placeholder.write("Training complete.")

    # Final test evaluation
    test_loss, test_acc = model.evaluate(x_test, test_labels, verbose=0)
    st.success(f"Test accuracy: {test_acc:.4f}")

