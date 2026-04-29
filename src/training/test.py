"""
"""
import logging
from pathlib import Path
import json

import torch
import torch.nn as nn

from utils.data import get_dataloader
from engine import eval_one_epoch
from utils.metrics import compute_class_metrics, compute_advanced_metrics

logger = logging.getLogger(__name__)


def run_test(
        cfg: dict,
        model: nn.Module,
    ) -> dict:
    seed = cfg.get("seed", None)
    test_dataloader = get_dataloader(cfg["data"], seed=seed)["test_loader"]
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = eval_one_epoch(
        dataloader=test_dataloader,
        model=model,
        criterion=criterion,
        device=device,
        )
    
    if cfg["eval"].get("advanced_metrics", False):
        logger.info("Advanced metrics are active")
        targets = metrics["targets"]
        preds = metrics["preds"]

        macro_metrics = compute_advanced_metrics(targets=targets, preds=preds)
        class_metrics = compute_class_metrics(targets=targets, preds=preds)
    
        test_metrics = metrics | macro_metrics | class_metrics
    
    else:
        test_metrics = metrics

    save_dir = cfg.get("save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_path = save_dir / "test_metrics.json"
        with open(save_path, "w") as f:
            json.dump(test_metrics, f, indent=1, default=lambda x: x.tolist())
        logger.info(f"Test metrics saved to {save_dir}")

    return test_metrics