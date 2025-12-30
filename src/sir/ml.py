"""ML/DL models for SIR parameter inference.

Defines multiple Keras architectures (MLP/CNN/TCN/RNN/Transformer and
distributional variants) plus training and inference timing helpers. These
models map I(t) to (beta, gamma) and are used by benchmark scripts for ML
baselines. Requires TensorFlow to be installed.
"""


from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Dict, Optional, Sequence, Tuple, Union
import time

import numpy as np

from src.sir.io import ensure_dir

@dataclass
class TrainResult:
    history: object
    eval_metrics: Dict[str, float]
    train_time_sec: float


def _require_tf():
    try:
        import tensorflow as tf

        from src.sir.config import configure_tensorflow_memory_growth

        configure_tensorflow_memory_growth(tf)
        return tf
    except Exception as exc:
        # Fail fast if TensorFlow is not available.
        raise RuntimeError("TensorFlow is required for ML experiments") from exc


def _compile_standard(model: "object") -> None:
    tf = _require_tf()
    from tensorflow.keras.optimizers import AdamW

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        loss=tf.keras.losses.Huber(delta=1e-4),
        metrics=[tf.keras.metrics.R2Score()],
    )


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


def build_linear(input_dim: int = 1001, l2: float = 1e-4) -> "object":
    """Linear regression baseline (Dense(2)) with optional L2 regularization."""
    tf = _require_tf()
    from tensorflow.keras import regularizers

    inputs = tf.keras.Input(shape=(input_dim,))
    outputs = tf.keras.layers.Dense(
        2,
        activation="linear",
        kernel_regularizer=regularizers.l2(l2),
    )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_resmlp(
    input_dim: int = 1001,
    width: int = 128,
    depth: int = 4,
    dropout: float = 0.1,
) -> "object":
    """Residual MLP with LayerNorm blocks."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(width, activation=swish)(inputs)
    for _ in range(depth):
        shortcut = x
        x = tf.keras.layers.Dense(width, activation=swish)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Dense(width)(x)
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(swish)(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_tcn(
    input_len: int = 1001,
    filters: int = 64,
    kernel_size: int = 3,
    dilations: Sequence[int] = (1, 2, 4, 8, 16),
    dropout: float = 0.1,
) -> "object":
    """Temporal Convolutional Network (dilated residual Conv1D)."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    for d in dilations:
        residual = x
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=d,
            activation=swish,
        )(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=d,
        )(x)
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)
        x = tf.keras.layers.Add()([residual, x])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(swish)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_inception(
    input_len: int = 1001,
    filters: int = 32,
    kernel_sizes: Sequence[int] = (3, 5, 9),
    depth: int = 3,
) -> "object":
    """Inception-style 1D CNN with multi-kernel blocks."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    for _ in range(depth):
        residual = x
        branches = [
            tf.keras.layers.Conv1D(filters, k, padding="same", activation=swish)(x)
            for k in kernel_sizes
        ]
        x = tf.keras.layers.Concatenate()(branches)
        x = tf.keras.layers.Conv1D(filters, 1, padding="same", activation=swish)(x)
        if residual.shape[-1] != x.shape[-1]:
            residual = tf.keras.layers.Conv1D(int(x.shape[-1]), 1, padding="same")(residual)
        x = tf.keras.layers.Add()([residual, x])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_attn_cnn(
    input_len: int = 1001,
    filters: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.1,
) -> "object":
    """Conv1D feature extractor with attention pooling."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation=swish)(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation=swish)(x)

    scores = tf.keras.layers.Dense(1)(x)
    weights = tf.keras.layers.Softmax(axis=1)(scores)
    x = tf.keras.layers.Multiply()([x, weights])
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_gru(
    input_len: int = 1001,
    units: int = 64,
    bidirectional: bool = False,
) -> "object":
    """GRU-based sequence encoder."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.layers import GRU, Bidirectional

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    rnn = GRU(units)
    if bidirectional:
        x = Bidirectional(rnn)(x)
    else:
        x = rnn(x)
    x = tf.keras.layers.Dense(units, activation=swish)(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_lstm(
    input_len: int = 1001,
    units: int = 64,
    bidirectional: bool = False,
) -> "object":
    """LSTM-based sequence encoder."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.layers import LSTM, Bidirectional

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    rnn = LSTM(units)
    if bidirectional:
        x = Bidirectional(rnn)(x)
    else:
        x = rnn(x)
    x = tf.keras.layers.Dense(units, activation=swish)(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_conv_gru(
    input_len: int = 1001,
    filters: int = 32,
    kernel_size: int = 5,
    stride: int = 2,
    units: int = 64,
) -> "object":
    """Conv1D downsampling followed by GRU."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.layers import GRU

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding="same", activation=swish)(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size, strides=stride, padding="same", activation=swish)(x)
    x = GRU(units)(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def build_transformer(
    input_len: int = 1001,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    depth: int = 2,
    dropout: float = 0.1,
) -> "object":
    """Small Transformer encoder with pooling."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.layers import MultiHeadAttention

    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    key_dim = d_model // num_heads

    inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(inputs)
    x = tf.keras.layers.Dense(d_model)(x)

    positions = tf.range(start=0, limit=input_len, delta=1, dtype=tf.int32)
    pos_embed = tf.keras.layers.Embedding(input_len, d_model)(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = tf.keras.layers.Add()([x, pos_embed])

    for _ in range(depth):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
        x = tf.keras.layers.Add()([x, attn_out])
        x = tf.keras.layers.LayerNormalization()(x)
        ff = tf.keras.layers.Dense(ff_dim, activation=swish)(x)
        ff = tf.keras.layers.Dropout(dropout)(ff)
        ff = tf.keras.layers.Dense(d_model)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(2, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    _compile_standard(model)
    model.sir_output = "mean"
    return model


def _heteroscedastic_nll(y_true, y_pred):
    tf = _require_tf()
    mu = y_pred[..., :2]
    log_var = tf.clip_by_value(y_pred[..., 2:], -10.0, 10.0)
    var = tf.exp(log_var)
    nll = 0.5 * (log_var + tf.square(y_true - mu) / (var + 1e-8))
    return tf.reduce_mean(tf.reduce_sum(nll, axis=-1))


def build_mlp_heteroscedastic(
    input_dim: int = 1001,
    width: int = 128,
    depth: int = 4,
    dropout: float = 0.1,
) -> "object":
    """MLP that predicts mean and log-variance for (beta, gamma)."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.optimizers import AdamW

    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(depth):
        x = tf.keras.layers.Dense(width, activation=swish)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(4, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        loss=_heteroscedastic_nll,
    )
    model.sir_output = "heteroscedastic"
    model.sir_monitor = "val_loss"
    return model


def _mdn_nll(n_components: int, n_dims: int = 2):
    def loss(y_true, y_pred):
        tf = _require_tf()
        k = n_components
        pi_logits = y_pred[:, :k]
        mu = y_pred[:, k : k + k * n_dims]
        log_sigma = y_pred[:, k + k * n_dims :]

        mu = tf.reshape(mu, (-1, k, n_dims))
        log_sigma = tf.reshape(log_sigma, (-1, k, n_dims))
        log_sigma = tf.clip_by_value(log_sigma, -10.0, 10.0)
        sigma = tf.exp(log_sigma)

        y = tf.expand_dims(y_true, axis=1)
        log_prob = -0.5 * (
            tf.square((y - mu) / (sigma + 1e-8))
            + 2.0 * log_sigma
            + tf.math.log(2.0 * np.pi)
        )
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        log_pi = tf.nn.log_softmax(pi_logits, axis=-1)
        log_sum = tf.reduce_logsumexp(log_pi + log_prob, axis=-1)
        return -tf.reduce_mean(log_sum)

    return loss


def build_mlp_mdn(
    input_dim: int = 1001,
    n_components: int = 3,
    width: int = 128,
    depth: int = 4,
    dropout: float = 0.1,
) -> "object":
    """Mixture Density Network predicting a Gaussian mixture over (beta, gamma)."""
    tf = _require_tf()
    from tensorflow.keras.activations import swish
    from tensorflow.keras.optimizers import AdamW

    output_dim = n_components + n_components * 2 * 2
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(depth):
        x = tf.keras.layers.Dense(width, activation=swish)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(output_dim, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5),
        loss=_mdn_nll(n_components),
    )
    model.sir_output = "mdn"
    model.sir_mdn_components = n_components
    model.sir_monitor = "val_loss"
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

    monitor = getattr(model, "sir_monitor", "val_r2_score")
    # Early stopping for reproducible training runs.
    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
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
    if not isinstance(eval_out, (list, tuple)):
        eval_out = [eval_out]
    metrics = dict(zip(model.metrics_names, eval_out))

    return TrainResult(history=history, eval_metrics=metrics, train_time_sec=train_time)


def _model_summary_text(model: "object") -> str:
    """Capture model.summary() as text."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def _can_plot_model(tf: "object") -> bool:
    """Return True if plot_model can run (pydot + graphviz available)."""
    is_graphviz = getattr(tf.keras.utils, "is_graphviz_available", None)
    is_pydot = getattr(tf.keras.utils, "is_pydot_available", None)
    if callable(is_graphviz) and callable(is_pydot):
        return bool(is_graphviz()) and bool(is_pydot())
    try:
        import pydot  # noqa: F401
    except Exception:
        return False
    return shutil.which("dot") is not None


def save_model_artifacts(
    model: "object",
    name: str,
    out_dir: Union[Path, str],
) -> Dict[str, object]:
    """Save weights and architecture visualization for a trained model."""
    tf = _require_tf()
    logger = logging.getLogger(__name__)

    out_dir = Path(out_dir)
    model_dir = out_dir / name
    ensure_dir(model_dir)

    artifacts: Dict[str, object] = {
        "name": name,
        "model_dir": model_dir,
        "weights_prefix": None,
        "weights_file": None,
        "weights_files": [],
        "architecture_plot": None,
        "architecture_json": None,
        "summary_txt": None,
    }

    # Save weights in a Keras-compatible filename.
    weights_path = model_dir / f"{name}.weights.h5"
    try:
        model.save_weights(str(weights_path))
        artifacts["weights_prefix"] = weights_path
        artifacts["weights_file"] = weights_path
        artifacts["weights_files"] = [weights_path]
    except Exception as exc:
        logger.warning("Failed to save weights for %s: %s", name, exc)

    # Persist JSON architecture for reproducibility.
    try:
        arch_path = model_dir / "architecture.json"
        arch_path.write_text(model.to_json(), encoding="utf-8")
        artifacts["architecture_json"] = arch_path
    except Exception as exc:
        logger.warning("Failed to save architecture JSON for %s: %s", name, exc)

    # Persist a text summary for quick inspection.
    try:
        summary_path = model_dir / "summary.txt"
        summary_path.write_text(_model_summary_text(model), encoding="utf-8")
        artifacts["summary_txt"] = summary_path
    except Exception as exc:
        logger.warning("Failed to save summary for %s: %s", name, exc)

    # Render a model diagram if graphviz/pydot are available.
    if _can_plot_model(tf):
        try:
            plot_path = model_dir / "architecture.png"
            tf.keras.utils.plot_model(
                model,
                to_file=str(plot_path),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
                dpi=150,
            )
            artifacts["architecture_plot"] = plot_path
        except Exception as exc:
            logger.warning("Failed to plot model for %s: %s", name, exc)
    else:
        logger.warning(
            "Skipping plot_model for %s (Graphviz + pydot not available)",
            name,
        )

    logger.info("Saved model artifacts for %s to %s", name, model_dir)
    return artifacts


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


def predict_params(model: "object", X: np.ndarray) -> np.ndarray:
    """Predict (beta, gamma) for models with different output heads."""
    y_raw = model.predict(X, verbose=0)
    output_type = getattr(model, "sir_output", "mean")
    if output_type == "heteroscedastic":
        return np.asarray(y_raw)[:, :2]
    if output_type == "mdn":
        y_raw = np.asarray(y_raw)
        k = int(getattr(model, "sir_mdn_components", 1))
        pi_logits = y_raw[:, :k]
        mu = y_raw[:, k : k + k * 2]
        mu = mu.reshape(-1, k, 2)
        # Softmax over mixture weights.
        exp_logits = np.exp(pi_logits - np.max(pi_logits, axis=1, keepdims=True))
        weights = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.sum(mu * weights[:, :, None], axis=1)
    return np.asarray(y_raw)
