"""
Utilities for plotting metrics computed during training or evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def plot_loss(
        train_loss: list[float],
        val_loss: list[float] | None = None,
        figsize: tuple[int] = (8, 5),
        save_path: str | None = None,
    ) -> None:
    """
    Plot training and validation loss over epochs.

    Parameters
    ----------
    train_loss : list of float
        Training loss values, one per epoch.
    val_loss : list of float or None, optional
        Validation loss values, one per epoch. If ``None``, only the
        training loss is plotted.
    figsize : tuple of int, default=(8,5)
        Width and height of the figure in inches.
    save_path : str or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """    
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_loss, label="Train Loss")
    if val_loss is not None:
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
        val_acc: list[float],
        figsize: tuple[int] = (8, 5),
        save_path: str | None = None,
    ) -> None:
    """
    Plot training and validation accuracy over epochs.
 
    Parameters
    ----------
    train_acc : list of float
        Training accuracy values, one per epoch.
    val_acc : list of float or None, optional
        Validation accuracy values, one per epoch. If ``None``, only the
        training accuracy is plotted.
    figsize : tuple of int, default=(8,5)
        Width and height of the figure in inches.
    save_path : str or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    epochs = range(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, train_acc, label="Train Accuracy")
    if val_acc is not None:
        ax.plot(epochs, val_acc, label="Val Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy over epochs")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    _show_or_save(fig, save_path=save_path)


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list[str] | None = None,
        save_path: str | None = None,
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
    save_path : str or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is displayed interactively and not saved.
    """
    figsize = (max(6, len(confusion_matrix)), max(5, len(confusion_matrix) - 1))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        confusion_matrix,
        cmap="Blues",
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


def _show_or_save(fig: Figure, save_path: str | None = None, dpi: int = 150) -> None:
    """
    Display a figure interactively or save it to disk.
 
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to display or save.
    save_path : str or None, optional
        File path where the figure will be saved. If ``None``, the figure
        is shown interactively via ``plt.show()``.
    dpi : int, default=150
        Resolution in dots per inch used when saving. Has no effect when
        displaying interactively.
    """

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()