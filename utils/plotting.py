"""
Utilities for plotting metrics computed during training or evaluation.
"""
from pathlib import Path
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_loss(
        train_loss: list[float],
        val_loss: list[float] | None = None,
        figsize: tuple[int] = (8, 5),
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot training (and validation) loss over epochs.

    Parameters
    ----------
    train_loss : list of float
        Training loss values, one per epoch.
    val_loss : list of float or None, optional
        Validation loss values, one per epoch. If ``None``, only the
        training loss is plotted.
    figsize : tuple of int, default=(8,5)
        Width and height of the figure in inches.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting training loss")
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
        logger.info("Plotting validation loss")
        ax.plot(epochs, val_loss, label="Val Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over epochs")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_accuracy(
        train_acc: list[float],
        val_acc: list[float] | None = None,
        figsize: tuple[int] = (8, 5),
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot training (and validation) accuracy over epochs.
 
    Parameters
    ----------
    train_acc : list of float
        Training accuracy values, one per epoch.
    val_acc : list of float or None, optional
        Validation accuracy values, one per epoch. If ``None``, only the
        training accuracy is plotted.
    figsize : tuple of int, default=(8,5)
        Width and height of the figure in inches.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting training accuracy")
    epochs = range(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_acc, label="Train Accuracy")
    if val_acc is not None:
        logger.info("Plotting validation accuracy")
        ax.plot(epochs, val_acc, label="Val Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy over epochs")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_training_curves(
        train_loss: list[float],
        train_acc: list[float],
        val_loss: list[float] | None = None,
        val_acc: list[float] | None = None,
        figsize: tuple[int] = (14,5),
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot training (and validation) loss and accuracy over epochs
    side by side in one figure.

    Parameters
    ----------
    train_loss : list of float
        Training loss values, one per epoch.
    train_acc : list of float
        Training accuracy values, one per epoch.
    val_loss : list of float or None, optional
        Validation loss values, one per epoch. If ``None``, only the
        training loss is plotted.
    val_acc : list of float or None, optional
        Validation accuracy values, one per epoch. If ``None``, only the
        training accuracy is plotted.
    figsize : tuple of int, default=(14,5)
        Width and height of the figure in inches.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting training curves")
    epochs = range(1, len(train_loss) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize)

    ax_loss.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
        logger.info("Plotting validation curves")
        ax_loss.plot(epochs, val_loss, label="Val Loss")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss over epochs")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(epochs, train_acc, label="Train Accuracy")
    if val_acc is not None:
        ax_acc.plot(epochs, val_acc, label="Val Accuracy")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_title("Accuracy over epochs")
    ax_acc.legend()
    ax_acc.grid(True)

    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_macro_advanced_metrics(
        precision_score: list[float],
        recall_score: list[float],
        f1_score: list[float],
        figsize: tuple[int] = (8, 5),
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot macro-averaged advanced metrics (precision, recall and F1) over epochs.

    Parameters
    ----------
    precision_score : list of float
        Macro-averaged precision values, one per epoch.
    recall_score : list of float
        Macro-averaged score values, one per epoch.
    f1_score : list of float
        Macro-averaged F1 values, one per epoch.
    figsize : tuple of int, default=(8,5)
        Width and height of the figure in inches.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting macro averaged advanced metrics (precision, recall and F1)")
    epochs = range(1, len(precision_score) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, precision_score, label="Precision")
    ax.plot(epochs, recall_score, label="Recall")
    ax.plot(epochs, f1_score, label="F1")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Macro-averaged advanced metrics over epochs")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_class_metrics(
        precision_score: list[float],
        recall_score: list[float],
        f1_score: list[float],
        class_names: list[str] | None = None,
        width : float = 0.25,
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot grouped bar chart of precision, recall and F1 score for each class.

    Function expects only one epoch. Use the final one.

    Parameters
    ----------
    precision_score : list of float
        Precision value for each class.
    recall_score : list of float
        Recall value for each class.
    f1_score : list of float
        F1 value for each class.
    class_names : list of str or None, optional
        Labels for each class, used as labels. If ``None``, integer indices
        are used.
    width : float, default=0.25
        Width of each bar. Grouped bar width is ``3 * width``.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting per class advanced metrics (precision, recall and F1)")
    n_classes = len(precision_score)
    x = range(n_classes)
    labels = class_names or [str(i) for i in x]

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 1.2), 5))
    ax.bar(x - width, precision_score, width=width, label="Precision", color="steelblue")
    ax.bar(x, recall_score, width=width, label="Recall", color="darkorange")
    ax.bar(x + width, f1_score, width=width, label="F1", color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(labels=labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Metrics")
    ax.legend()
    ax.bar_label(ax.containers[2], fmt="%.2f", padding=2, fontsize=8)  # F1 labels
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list[str] | None = None,
        cmap: str = "flare",
        save_path: str | Path | None = None,
    ) -> None:
    """
    Plot a confusion matrix as an annotated heatmap.
 
    Figure size is scaled automatically based on the number of classes.
    Intended to be used on final evaluation results, not per epoch.
 
    Parameters
    ----------
    confusion_matrix : np.ndarray
        Square matrix of shape ``(n_classes, n_classes)`` where entry
        ``[i, j]`` is the number of samples with true class ``i``
        predicted as class ``j``.
    class_names : list of str or None, optional
        Labels for each class, used as tick labels on both axes. If
        ``None``, integer indices are used.
    cmap : str, default="flare"
        Color of the heatmap.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    logger.info("Plotting confusion matrix")
    figsize = (max(6, len(confusion_matrix)), max(6, len(confusion_matrix)))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        cmap=cmap,
        annot=True,
        fmt="d",
        xticklabels=class_names or "auto",
        yticklabels=class_names or "auto",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    plt.tight_layout()
    _show_or_save(fig=fig, save_path=save_path)


def _show_or_save(
        fig: Figure,
        save_path: str | Path | None = None,
        dpi: int = 150,
    ) -> None:
    """
    Display a figure interactively or save it to disk.
 
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to display or save.
    save_path : str or Path or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is shown interactively via ``plt.show()``.
    dpi : int, default=150
        Resolution in dots per inch used when saving. Has no effect when
        displaying interactively.
    """

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Figure saved")
    else:
        logger.info("Displaying figure")
        plt.show()