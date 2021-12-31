from dataclasses import dataclass, field, asdict
import pandas as pd
import pathlib
from typing import Any, Dict, List
from config import config
import wandb


@dataclass
class FilePaths:
    """Class to keep track of the files."""

    train_images: pathlib.Path = pathlib.Path(config.DATA_DIR, "train")
    test_images: pathlib.Path = pathlib.Path(config.DATA_DIR, "test")
    train_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "raw/train.csv")
    test_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "raw/test.csv")
    sub_csv: pathlib.Path = pathlib.Path(
        config.DATA_DIR,
        "raw/sample_submission.csv",
    )
    folds_csv: pathlib.Path = pathlib.Path(
        config.DATA_DIR,
        "processed/train.csv",
    )
    weight_path: pathlib.Path = pathlib.Path(config.MODEL_DIR)
    oof_csv: pathlib.Path = pathlib.Path(config.DATA_DIR, "processed")
    wandb_dir: pathlib.Path = pathlib.Path(config.WANDB_DIR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataLoaderParams:
    """Class to keep track of the data loader parameters."""

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": True,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    test_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            "shuffle": False,
            "collate_fn": None,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def get_len_train_loader(self) -> int:
        """Returns the length of the train loader.

        This is useful when using OneCycleLR.

        Returns:
            int(len_of_train_loader) (int): Length of the train loader.
        """
        total_rows = pd.read_csv(FilePaths().train_csv).shape[
            0
        ]  # get total number of rows/images
        total_rows_per_fold = total_rows / (MakeFolds().num_folds)
        total_rows_per_training = total_rows_per_fold * (
            MakeFolds().num_folds - 1
        )  # if got 1000 images, 10 folds, then train on 9 folds = 1000/10 * (10-1) = 100 * 9 = 900
        len_of_train_loader = (
            total_rows_per_training // self.train_loader["batch_size"]
        )  # if 900 rows, bs is 16, then 900/16 = 56.25, but we drop last if dataloader, so become 56 steps. if not 57 steps.
        return int(len_of_train_loader)


@dataclass
class MakeFolds:
    """A class to keep track of cross-validation schema.

    seed (int): random seed for reproducibility.
    num_folds (int): number of folds.
    cv_schema (str): cross-validation schema.
    class_col_name (str): name of the target column.
    image_col_name (str): name of the image column.
    folds_csv (str): path to the folds csv.
    """

    seed: int = 199294
    num_folds: int = 5
    cv_schema: str = "StratifiedKFold"
    class_col_name: str = "Pawpularity"
    image_col_name: str = "Id"
    image_extension: str = ".jpg"  # ".jpg"
    use_sturge: bool = True
    is_normalize: bool = True
    folds_csv: pathlib.Path = FilePaths().folds_csv

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AugmentationParams:
    """Class to keep track of the augmentation parameters."""

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    image_size: int = 224
    mixup: bool = False
    mixup_params: Dict[str, Any] = field(
        default_factory=lambda: {"mixup_alpha": 0.5, "use_cuda": True}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CriterionParams:
    """A class to track loss function parameters."""

    train_criterion_name: str = "BCEWithLogitsLoss"
    valid_criterion_name: str = "BCEWithLogitsLoss"
    train_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "reduction": "mean",
            "pos_weight": None,
        }
    )
    valid_criterion_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "weight": None,
            "reduction": "mean",
            "pos_weight": None,
        }
    )


@dataclass
class ModelParams:
    """A class to track model parameters.

    model_name (str): name of the model.
    pretrained (bool): If True, use pretrained model.
    input_channels (int): RGB image - 3 channels or Grayscale 1 channel
    output_dimension (int): Final output neuron.
                      It is the number of classes in classification.
                      Caution: If you use sigmoid layer for Binary, then it is 1.
    classification_type (str): classification type.
    """

    model_name: str = "swin_base_patch4_window7_224"  # Debug
    pretrained: bool = True
    input_channels: int = 3
    output_dimension: int = 1
    classification_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def check_dimension(self) -> None:
        """Check if the output dimension is correct."""
        if (
            self.classification_type == "binary"
            and CriterionParams().train_criterion_name == "BCEWithLogitsLoss"
        ):
            assert self.output_dimension == 1, "Output dimension should be 1"
        elif self.classification_type == "multilabel":
            config.logger.info(
                "Check on output dimensions as we are likely using BCEWithLogitsLoss"
            )


@dataclass
class GlobalTrainParams:
    debug: bool = True
    debug_multipler: int = 16
    epochs: int = 1  # 1 or 2 when debug
    use_amp: bool = True
    mixup: bool = AugmentationParams().mixup
    patience: int = 2
    model_name: str = ModelParams().model_name
    num_classes: int = ModelParams().output_dimension
    classification_type: str = ModelParams().classification_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class OptimizerParams:
    """A class to track optimizer parameters.

    optimizer_name (str): name of the optimizer.
    lr (float): learning rate.
    weight_decay (float): weight decay.
    """

    optimizer_name: str = "AdamW"
    optimizer_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 3e-4,
            "betas": (0.9, 0.999),
            "amsgrad": False,
            "weight_decay": 1e-3,
            "eps": 1e-08,
        }
    )
    # 1e-3 when debug mode else 3e-4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SchedulerParams:
    """A class to track Scheduler Params."""

    scheduler_name: str = "OneCycleLR"  # Debug
    if scheduler_name == "CosineAnnealingWarmRestarts":

        scheduler_params: Dict[str, Any] = field(
            default_factory=lambda: {
                "T_0": 10,
                "T_mult": 1,
                "eta_min": 1e-6,
                "last_epoch": -1,
            }
        )
    elif scheduler_name == "OneCycleLR":
        scheduler_params: Dict[str, Any] = field(
            default_factory=lambda: {
                "max_lr": 3e-5,
                "steps_per_epoch": DataLoaderParams().get_len_train_loader(),
                "epochs": GlobalTrainParams().epochs,
                "pct_start": 0.3,
                "anneal_strategy": "cos",
                "div_factor": 25,  # default is 25
                "three_phase": False,
                "last_epoch": -1,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class WandbParams:
    """A class to track wandb parameters."""

    project: str = "Petfinder"
    entity: str = "reighns"
    save_code: bool = True
    job_type: str = "Train"
    # add an unique group id behind group name.
    group: str = f"{GlobalTrainParams().model_name}_{MakeFolds().num_folds}_folds_{wandb.util.generate_id()}"
    dir: str = FilePaths().wandb_dir

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
