"""
Optimizer factory for training neural networks.
"""
import torch.optim as optim
import torch.nn as nn


def get_optim(cfg: dict, model: nn.Module) -> optim.Optimizer:
    """
    Build and return an optimizer configured from a dictionary.

    Parameters
    ----------
    cfg : dict
        Optimizer configuration. Expected keys:

        - ``name`` : str
            Optimizer name. Supported values are ``sgd``, ``adam`` or ``adamw``.
        - ``lr`` : float, default=0.001
            Learning rate. 
        - ``weight_decay`` : float, optional
            L2 regularization. Default is ``0.0`` for ``sgd`` and ``adam`` or
            ``0.01`` for ``adamw``.
        - ``momentum`` : float, default=0.9
            Momentum factor, used only for ``sgd``.

    model : nn.Module
        Model whose parameters will be optimized.

    Returns
    -------
    optim.Optimizer
        Configured optimizer instance.

    Raises
    ------
    ValueError
        If ``name`` is missing from ``cfg`` or is not one of the supported
        values: ``sgd``, ``adam``, ``adamw``.

    Examples
    --------
    >>> optimizer = get_optim({"name": "adamw", "lr": 3e-4}, model)
    """
    if "name" not in cfg:
        raise ValueError("Optimizer configuration must include a 'name' key.")
    name = cfg["name"]
    
    lr = cfg.get("lr", 0.001)

    match name:
        case "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=cfg.get("momentum", 0.9),
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        
        case "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=cfg.get("weight_decay", 0.0),
            )

        case "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=cfg.get("weight_decay", 0.01),
            )
        
        case _:
            raise ValueError(
                f"Unknown optimizer: '{name}'. "
                f"Optimizers available: ['sgd', 'adam', 'adamw']"
            )
        
    return optimizer