"""
Utilities for plotting metrics computed during training or evaluation
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
        List fo the validation accuracy of the NN over epochs. Default is None.
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



def _show_or_save(fig: Figure, save_path: str | None = None, dpi: int = 150) -> None:
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()