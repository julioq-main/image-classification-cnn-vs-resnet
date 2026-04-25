"""
Utilities for plotting metrics computed during training or evaluation
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
    Make a plot of the training (and validation) loss over epochs.

    Parameters
    ----------
    train_loss : list[float]
        List of the training loss of the NN over epochs.
    val_loss : list[float], optional
        List fo the validation loss of the NN over epochs. Default is None.
    fig_size : tuple[int]
        Tuple containing the size of the figure going to be plotted.
        Default is (8, 5).
    save_path : str, optional
        Path to save the plot. Default is None
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

    _show_or_save(fig, save_path=save_path)


def plot_accuracy(
        train_acc: list[float],
        val_acc: list[float],
        figsize: tuple[int] = (8, 5),
        save_path: str | None = None,
    ) -> None:
    """
    Make a plot of the training (and validation) accuracy over epochs.

    Parameters
    ----------
    train_acc : list[float]
        List of the training accuracy of the NN over epochs.
    val_acc : list[float], optional
        List of the validation accuracy of the NN over epochs. Default is None.
    fig_size : tuple[int]
        Tuple containing the size of the figure going to be plotted.
        Default is (8, 5).
    save_path : str, optional
        Path to save the plot. Default is None
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

    _show_or_save(fig, save_path=save_path)


def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        class_names: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
    """
    Make a heatmap of a confusion matrix. Intended to use on final epoch only.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        NdArray of the confusion matrix.
    class_names : list[str], optional
        List containing the name of each class. Default is None
    save_path : str, optional
        Path to save the plot. Default is None
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
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()