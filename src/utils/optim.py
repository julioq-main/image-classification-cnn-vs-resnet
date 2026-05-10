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

def get_scheduler(
        cfg: dict | None,
        optimizer: optim.Optimizer,
    ) -> optim.lr_scheduler.LRScheduler | None:
    """
    Build and return a learning rate scheduler configured from a dictionary.

    Parameters
    ----------
    cfg : dict or None
        Scheduler configuration. If ``None``, no scheduler is applied and
        the function returns ``None``. Expected keys:

        - ``name`` : str
            Scheduler name. Supported values are ``step`` and ``cosine``.

        For ``step``:

        - ``step_size`` : int
            Number of epochs between each learning rate decay step.
        - ``gamma`` : float, default=0.1
            Multiplicative factor applied to the learning rate at each step.

        For ``cosine``:

        - ``T_max`` : int
            Number of epochs for one cosine annealing cycle. Typically set
            to the total number of training epochs.
        - ``eta_min`` : float, default=0
            Minimum learning rate at the end of the cycle.

    optimizer : optim.Optimizer
        Optimizer whose learning rate will be scheduled. Must be fully
        configured before being passed to this function.

    Returns
    -------
    optim.lr_scheduler.LRScheduler or None
        Configured scheduler instance, or ``None`` if ``cfg`` is ``None``.

    Raises
    ------
    ValueError
        If ``name`` is not one of the supported scheduler names.

    Examples
    --------
    >>> scheduler = get_scheduler({"name": "cosine", "T_max": 50}, optimizer)
    >>> scheduler = get_scheduler({"name": "step", "step_size": 10}, optimizer)
    >>> scheduler = get_scheduler(None, optimizer)  # no scheduling
    """
    if cfg is None:
        return None
    
    name = cfg["name"]

    match name:
        case "step":
            step_size = cfg["step_size"]
            gamma = cfg.get("gamma", 0.1)
            
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=step_size,
                gamma=gamma,
                )
            
        case "cosine":
            T_max = cfg["T_max"]
            eta_min=cfg.get("eta_min", 0)

            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
        
        case _:
            raise ValueError(
                f"Unknown LR Scheduler: '{name}'. "
                f"Schedulers available: ['step', 'cosine']"
            )
        
    return lr_scheduler