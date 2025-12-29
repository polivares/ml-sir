"""ML/DL models for SIR parameter inference.

Defines Keras MLP and CNN1D architectures plus training and inference timing
helpers. These models map I(t) to (beta, gamma) and are used by benchmark
scripts for ML baselines. Requires TensorFlow to be installed.
"""


from dataclasses import dataclass
from typing import Dict, Tuple
import time

import numpy as np


@dataclass
class TrainResult:
    history: object
    eval_metrics: Dict[str, float]
    train_time_sec: float


def _require_tf():
    try:
        import tensorflow as tf

        return tf
    except Exception as exc:
        # Fail fast if TensorFlow is not available.
        raise RuntimeError("TensorFlow is required for ML experiments") from exc


def build_mlp(input_dim: int = 1001) -> "object":
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.optimizers import AdamW

    # MLP baseline used in current notebooks.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(256, activation=swish))
    model.add(tf.keras.layers.Dense(128, activation=swish))
    model.add(tf.keras.layers.Dense(32, activation=swish))
    model.add(tf.keras.layers.Dense(16, activation=swish))
    model.add(tf.keras.layers.Dense(64, activation=swish))
    model.add(tf.keras.layers.Dense(16, activation=swish))
    # Final linear layer outputs [beta, gamma].
    model.add(tf.keras.layers.Dense(2, activation="linear"))

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        # Huber is robust to occasional outliers in targets.
        loss=tf.keras.losses.Huber(delta=1e-4),
        metrics=[tf.keras.metrics.R2Score()],
    )
    return model


def build_mlp_branched(input_dim: int = 1001) -> "object":
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.layers import Dense, Concatenate

    # Shared trunk with separate heads for beta and gamma.
    inputs = tf.keras.Input(shape=(input_dim,))
    x = Dense(256, activation=swish)(inputs)
    x = Dense(128, activation=swish)(x)
    x = Dense(32, activation=swish)(x)
    x = Dense(16, activation=swish)(x)

    b = Dense(64, activation=swish)(x)
    b = Dense(16, activation=swish)(b)
    b = Dense(1, activation="linear")(b)

    g = Dense(64, activation=swish)(x)
    g = Dense(16, activation=swish)(g)
    g = Dense(1, activation="linear")(g)

    outputs = Concatenate()([b, g])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        # Match the MLP loss/metrics for apples-to-apples comparison.
        loss=tf.keras.losses.Huber(delta=1e-4),
        metrics=[tf.keras.metrics.R2Score()],
    )
    return model


def build_cnn1d(input_len: int = 1001) -> "object":
    tf = _require_tf()
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer, Reshape, Conv1D, GlobalAveragePooling1D, Dense
    from tensorflow.keras.optimizers import AdamW
    from tensorflow.keras.activations import swish
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.metrics import R2Score

    # Lightweight CNN to capture local temporal patterns.
    model = Sequential([
        InputLayer(input_shape=(input_len,)),
        # Convert 1D series into (time, channels) for Conv1D.
        Reshape((input_len, 1)),
        Conv1D(filters=64, kernel_size=9, activation=swish, padding="same"),
        Conv1D(filters=64, kernel_size=5, activation=swish, padding="same"),
        GlobalAveragePooling1D(),
        Dense(64, activation=swish),
        Dense(2, activation="linear"),
    ])

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        # Same loss/metric as MLP to keep comparisons fair.
        loss=Huber(delta=1e-4),
        metrics=[R2Score()],
    )
    return model


def train_model(
    model: "object",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    patience: int = 15,
    batch_size: int = 32,
) -> TrainResult:
    tf = _require_tf()
    from tensorflow.keras.callbacks import EarlyStopping

    # Early stopping for reproducible training runs.
    early_stop = EarlyStopping(
        monitor="val_r2_score", patience=patience, restore_best_weights=True
    )
    # Time the full training loop for reporting.
    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )
    train_time = time.perf_counter() - start

    # Evaluate on validation to return a compact summary.
    eval_out = model.evaluate(X_val, y_val, verbose=0)
    metrics = dict(zip(model.metrics_names, eval_out))

    return TrainResult(history=history, eval_metrics=metrics, train_time_sec=train_time)


def predict_time_per_sample(model: "object", X: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Estimate per-sample inference time by sampling a few single predictions."""
    rng = np.random.default_rng(0)
    n = X.shape[0]
    # Sample a few single predictions to approximate per-sample latency.
    idx = rng.choice(n, size=min(n_samples, n), replace=False)
    times = []
    for i in idx:
        start = time.perf_counter()
        _ = model.predict(X[i : i + 1], verbose=0)
        times.append(time.perf_counter() - start)
    return np.asarray(times)
