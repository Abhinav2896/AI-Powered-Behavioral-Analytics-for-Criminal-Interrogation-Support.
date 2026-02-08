import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(timesteps=30, features=4, classes=5):
    x = layers.Input(shape=(timesteps, features))
    h = layers.Dense(32, activation="relu")(x)
    h = layers.LSTM(64, return_sequences=True)(h)
    h = layers.LSTM(64)(h)
    h = layers.Dense(64, activation="relu")(h)
    y = layers.Dense(classes, activation="softmax")(h)
    m = models.Model(inputs=x, outputs=y)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def load_dataset(x_path, y_path):
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y

def train_and_save(x_path, y_path, out_path):
    x, y = load_dataset(x_path, y_path)
    m = build_model(timesteps=x.shape[1], features=x.shape[2], classes=len(np.unique(y)))
    m.fit(x, y, epochs=20, batch_size=32, validation_split=0.2)
    m.save(out_path)

if __name__ == "__main__":
    x_path = os.environ.get("LIP_X_PATH", "lip_x.npy")
    y_path = os.environ.get("LIP_Y_PATH", "lip_y.npy")
    out = os.environ.get("LIP_OUT_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lip_state_model.h5"))
    train_and_save(x_path, y_path, out)
