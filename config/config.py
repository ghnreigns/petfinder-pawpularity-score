# Configurations
import logging
import warnings
from pathlib import Path
import sys

# import pretty_errors  # NOQA: F401 (imported but unused)
# from rich.logging import RichHandler
import torch

########################################################### Repository's Names ###########################################################
AUTHOR = "Hongnan G."
REPO = "reighns_pytorch_pipeline"

########################################################### Torch Device ###########################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################### Creating Directories ###########################################################

# This will create all the folders needed. Which begs the question on when should I execute this script?
BASE_DIR = Path(__file__).parent.parent.absolute()  # C:\Users\reigHns\mnist

CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
STORES_DIR = Path(BASE_DIR, "stores")


# Local stores
BLOB_STORE = Path(STORES_DIR, "blob")
FEATURE_STORE = Path(STORES_DIR, "feature")
MODEL_REGISTRY = Path(STORES_DIR, "model")
TENSORBOARD = Path(STORES_DIR, "tensorboard")
WANDB_DIR = Path(STORES_DIR, "wandb")

# Data folders # TODO: Uncomment if init a new project, otherwise, comment out as I want to put many competition data in the same folder.
# RAW_DATA_DIR = Path(DATA_DIR, "raw")
# PROCESSED_DATA_DIR = Path(DATA_DIR, "processed")
# TRAIN_DATA_DIR = Path(DATA_DIR, "train")
# TEST_DATA_DIR = Path(DATA_DIR, "test")


# Create dirs
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORES_DIR.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)
FEATURE_STORE.mkdir(parents=True, exist_ok=True)
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
WANDB_DIR.mkdir(parents=True, exist_ok=True)
# TODO: Uncomment if init a new project, otherwise, comment out as I want to put many competition data in the same folder.
# RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
# PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
# TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
# TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
# new folder
TENSORBOARD.mkdir(parents=True, exist_ok=True)


# MLFlow model registry: Note the filepath is file:////C:\Users\reigHns\mnist\stores\model

# mlflow.set_tracking_uri(uri="file://" + str(MODEL_REGISTRY.absolute()))
# workaround for windows at the moment
# TODO : Switch to Linux.
# mlflow.set_tracking_uri(uri="file://" + "C:/Users/reigHns/mnist/stores/model")

########################################################### Suppress User Warnings ###########################################################
warnings.filterwarnings("ignore", category=UserWarning)


def init_logger(
    log_file: str = Path(LOGS_DIR, "info.log"),
    module_name: Optional[str] = None,
    level=logging.INFO,
) -> logging.Logger:
    """Initialize logger and save to file.

    Consider having more log_file paths to save, eg: debug.log, error.log, etc.

    Args:
        log_file (str, optional): [description]. Defaults to Path(LOGS_DIR, "info.log").

    Returns:
        logging.Logger: [description]
    """

    if module_name is None:
        logger = logging.getLogger(__name__)
    else:
        # get module name, useful for multi-module logging
        logger = logging.getLogger(module_name)

    logger.setLevel(level)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
