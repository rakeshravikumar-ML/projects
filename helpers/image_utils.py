"""
utils/helpers.py

Professional, reusable helpers for CV/ML notebooks.
- Typed function signatures
- Clear, framework-agnostic docstrings
- Minimal side effects (plotters return fig/ax)
- Robust image decoding (png/jpg), consistent dtype
"""

from __future__ import annotations

import datetime as _dt
import itertools
import os
import zipfile
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

__all__ = [
    "load_and_prep_image",
    "make_confusion_matrix",
    "pred_and_plot",
    "create_tensorboard_callback",
    "plot_loss_curves",
    "compare_histories",
    "unzip_data",
    "walk_through_dir",
    "calculate_results",
]


# ---------- Data & I/O ----------

def load_and_prep_image(
    filename: str | Path,
    img_shape: int = 224,
    scale: bool = True,
    channels: int = 3,
) -> tf.Tensor:
    """
    Read an image file and return a float32 tensor resized to (img_shape, img_shape, channels).

    Parameters
    ----------
    filename : str | Path
        Path to the image on disk.
    img_shape : int, default=224
        Target spatial size (square). Use model's expected input size.
    scale : bool, default=True
        If True, scales pixel values to [0, 1]. Otherwise returns unscaled values.
    channels : int, default=3
        Number of color channels to decode to (3 = RGB).

    Returns
    -------
    tf.Tensor
        Image tensor of shape (img_shape, img_shape, channels), dtype=tf.float32.
    """
    filename = tf.convert_to_tensor(str(filename))
    img_bytes = tf.io.read_file(filename)
    # decode_image supports JPEG/PNG and infers type; force channels for consistency
    img = tf.image.decode_image(img_bytes, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0,1] float32
    img = tf.image.resize(img, [img_shape, img_shape], antialias=True)
    if not scale:
        img = img * 255.0  # back to [0,255] float32 if requested
    return img


def unzip_data(filename: str | Path, dest: str | Path | None = None) -> Path:
    """
    Unzip an archive into a destination directory.

    Parameters
    ----------
    filename : str | Path
        Path to the .zip file.
    dest : str | Path | None
        Destination directory. Defaults to the zip's parent directory.

    Returns
    -------
    Path
        Path to the destination directory.
    """
    filename = Path(filename)
    dest_path = Path(dest) if dest is not None else filename.parent
    dest_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(filename, "r") as zf:
        zf.extractall(dest_path)
    return dest_path


def walk_through_dir(dir_path: str | Path) -> List[Tuple[str, int, List[str]]]:
    """
    Walk a directory and summarize contents.

    Parameters
    ----------
    dir_path : str | Path
        Root directory to inspect.

    Returns
    -------
    list of tuples
        Each tuple: (directory_path, num_files, subdirectory_names).
        Also prints a concise summary for human reading.
    """
    dir_path = str(dir_path)
    summary: List[Tuple[str, int, List[str]]] = []
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        summary.append((dirpath, len(filenames), dirnames))
    return summary


# ---------- Visualization & Metrics ----------

def make_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (10, 10),
    text_size: int = 12,
    normalize: bool = False,
    cmap=plt.cm.Blues,
    title: str = "Confusion Matrix",
    savepath: Optional[str | Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a labeled confusion matrix comparing predictions to ground truth.

    Parameters
    ----------
    y_true, y_pred : Sequence[int]
        Ground-truth and predicted label indices.
    classes : Sequence[str] | None
        Optional class label names. If None, integer indices are used.
    figsize : (int, int)
        Matplotlib figure size.
    text_size : int
        Font size for cell annotations.
    normalize : bool
        If True, annotate with row-normalized percentages.
    cmap : Colormap
        Matplotlib colormap.
    title : str
        Figure title.
    savepath : str | Path | None
        If provided, saves the figure to this path.

    Returns
    -------
    (fig, ax) : Tuple[plt.Figure, plt.Axes]
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    labels = list(classes) if classes is not None else list(range(n_classes))

    if normalize:
        with np.errstate(all="ignore"):
            cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        annot = np.where(np.isfinite(cm_norm), (cm_norm * 100.0), 0.0)
        fmt = lambda v, r, c: f"{cm[r, c]} ({annot[r, c]:.1f}%)"
    else:
        fmt = lambda v, r, c: f"{cm[r, c]}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        ax.text(
            j,
            i,
            fmt(cm, i, j),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=text_size,
        )

    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    return fig, ax


def plot_loss_curves(
    history: tf.keras.callbacks.History,
    metric: str = "accuracy",
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot training/validation loss and a chosen metric from a Keras History.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Keras History returned by `model.fit(...)`.
    metric : str, default="accuracy"
        Metric key present in `history.history` (e.g., "accuracy", "auc").
    figsize : (int, int)
        Figure size.

    Returns
    -------
    (fig, ax)
    """
    h = history.history
    loss, val_loss = h.get("loss", []), h.get("val_loss", [])
    m, vm = h.get(metric, []), h.get(f"val_{metric}", [])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss, label="train_loss")
    if val_loss:
        ax.plot(val_loss, label="val_loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(loc="best")

    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.plot(m, label=f"train_{metric}")
    if vm:
        ax2.plot(vm, label=f"val_{metric}")
    ax2.set_title(metric.capitalize())
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="best")

    fig.tight_layout()
    fig2.tight_layout()
    return fig2, ax2  # return the metric figure/axes (most referenced)


def compare_histories(
    original_history: tf.keras.callbacks.History,
    new_history: tf.keras.callbacks.History,
    initial_epochs: int = 5,
    metric: str = "accuracy",
    figsize: Tuple[int, int] = (8, 8),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Compare two training phases (e.g., feature extraction vs fine-tuning).

    Parameters
    ----------
    original_history, new_history : History
        Keras History objects from consecutive training phases.
    initial_epochs : int
        Number of epochs in the original phase (vertical separator).
    metric : str
        Metric key to compare (e.g., "accuracy", "auc").
    figsize : (int, int)
        Figure size.

    Returns
    -------
    (fig, (ax1, ax2))
    """
    acc = list(original_history.history.get(metric, []))
    val_acc = list(original_history.history.get(f"val_{metric}", []))
    loss = list(original_history.history.get("loss", []))
    val_loss = list(original_history.history.get("val_loss", []))

    acc += list(new_history.history.get(metric, []))
    val_acc += list(new_history.history.get(f"val_{metric}", []))
    loss += list(new_history.history.get("loss", []))
    val_loss += list(new_history.history.get("val_loss", []))

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(acc, label=f"Training {metric}")
    ax1.plot(val_acc, label=f"Validation {metric}")
    ax1.axvline(x=initial_epochs - 1, linestyle="--", label="Start Fine-tuning")
    ax1.legend(loc="lower right")
    ax1.set_title(f"Training and Validation {metric.capitalize()}")

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(loss, label="Training Loss")
    ax2.plot(val_loss, label="Validation Loss")
    ax2.axvline(x=initial_epochs - 1, linestyle="--", label="Start Fine-tuning")
    ax2.legend(loc="upper right")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epoch")
    fig.tight_layout()
    return fig, (ax1, ax2)


def make_timestamped_logdir(dir_name: str | Path, experiment_name: str) -> Path:
    """Internal utility used by `create_tensorboard_callback`."""
    now = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(dir_name) / experiment_name / now
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def create_tensorboard_callback(
    dir_name: str | Path,
    experiment_name: str,
    profile_batch: str | int | Tuple[int, int] | None = None,
) -> tf.keras.callbacks.TensorBoard:
    """
    Create a TensorBoard callback with a timestamped log directory.

    Parameters
    ----------
    dir_name : str | Path
        Base logs directory.
    experiment_name : str
        Subdirectory name for this experiment.
    profile_batch : str | int | (int, int) | None
        Optional TensorBoard profiler setting (e.g., (5, 10)).

    Returns
    -------
    tf.keras.callbacks.TensorBoard
    """
    log_dir = make_timestamped_logdir(dir_name, experiment_name)
    print(f"Saving TensorBoard logs to: {log_dir}")
    return tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), profile_batch=profile_batch)


def calculate_results(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "weighted",
) -> Mapping[str, float]:
    """
    Compute accuracy, precision, recall, and F1 for (potentially) multiclass predictions.

    Parameters
    ----------
    y_true, y_pred : Sequence[int]
        Ground-truth and predicted label indices.
    average : str, default="weighted"
        Averaging scheme for precision/recall/F1 ("macro", "micro", "weighted").

    Returns
    -------
    dict
        { "accuracy": float, "precision": float, "recall": float, "f1": float }
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# ---------- Inference demo ----------

def pred_and_plot(
    model: tf.keras.Model,
    filename: str | Path,
    class_names: Sequence[str],
    img_shape: int = 224,
    show: bool = True,
) -> Tuple[str, np.ndarray, Optional[Tuple[plt.Figure, plt.Axes]]]:
    """
    Run a single-image prediction and (optionally) render it.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model with softmax/sigmoid output.
    filename : str | Path
        Image file to predict.
    class_names : Sequence[str]
        Ordered class names aligned to model outputs.
    img_shape : int, default=224
        Image resize target.
    show : bool, default=True
        If True, display the image with predicted class title.

    Returns
    -------
    (pred_class, probs, (fig, ax) or None)
        Predicted class name, probability vector (numpy), and figure/axes if plotted.
    """
    img = load_and_prep_image(filename, img_shape=img_shape, scale=True)
    logits = model.predict(tf.expand_dims(img, axis=0), verbose=0)
    probs = tf.nn.softmax(logits[0]) if logits.shape[-1] > 1 else tf.sigmoid(logits[0])
    probs = probs.numpy()

    if probs.size > 1:
        pred_idx = int(np.argmax(probs))
    else:
        pred_idx = int(np.round(probs[0]))
    pred_class = class_names[pred_idx]

    if show:
        fig, ax = plt.subplots()
        ax.imshow(img.numpy())
        ax.set_title(f"Prediction: {pred_class}")
        ax.axis("off")
        return pred_class, probs, (fig, ax)
    return pred_class, probs, None
