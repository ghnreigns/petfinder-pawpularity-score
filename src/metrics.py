from collections import defaultdict
import torch
from sklearn import metrics
from typing import List, Union, Dict
from config import config, global_params
import numpy as np
from torch.autograd.grad_mode import F
import torchmetrics
from torchmetrics.classification import auroc
import pandas as pd
import collections
from pathlib import Path
import torchmetrics

FOLDS = global_params.MakeFolds()
TRAIN_PARAMS = global_params.GlobalTrainParams()
LOGS_PARAMS = global_params.LogsParams()


metrics_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "metrics.log"),
    module_name="metrics",
)


class AverageLossMeter:
    """
    Computes and stores the average and current loss
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_batch_avg_loss = 0
        self.avg = 0
        self.running_total_loss = 0
        self.count = 0

    def update(self, curr_batch_avg_loss: float, batch_size: str):
        self.curr_batch_avg_loss = curr_batch_avg_loss
        self.running_total_loss += curr_batch_avg_loss * batch_size
        self.count += batch_size
        self.avg = self.running_total_loss / self.count


class MetricMonitor:
    """Monitor Metrics"""

    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(
            lambda: {"metric_score": 0, "count": 0, "average_score": 0}
        )

    def update(self, metric_name, metric_score):
        metric = self.metrics[metric_name]

        metric["metric_score"] += metric_score
        metric["count"] += 1
        metric["average_score"] = metric["metric_score"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["average_score"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def mse_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, is_rmse: bool = True
) -> torch.Tensor:
    """Compute r/mse score for regression.

    Args:
        y_true (torch.Tensor): True labels.
        y_pred (torch.Tensor): Predicted labels.
        is_rmse (bool): Whether return mse or rmse.

    Returns:
        squared_error (torch.Tensor): The squared error
    """

    squared = True if is_rmse is False else False
    squared_error = torchmetrics.MeanSquaredError(squared=squared)(
        y_pred, y_true
    )

    return squared_error
