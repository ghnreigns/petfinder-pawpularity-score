import collections
import glob
from pathlib import Path
from typing import Any, Dict, List, Union

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import config, global_params
from tqdm.auto import tqdm

from src import dataset, models, trainer, utils, metrics

MODEL = global_params.ModelParams()
FOLDS = global_params.MakeFolds()
LOADER_PARAMS = global_params.DataLoaderParams()
FILES = global_params.FilePaths()
WANDB = global_params.WandbParams()
device = config.DEVICE
# MODEL_ARTIFACTS_PATH = global_params.FilePaths().get_model_artifacts_path()
MODEL_ARTIFACTS_PATH = r"C:\Users\reighns\reighns_ml\kaggle\petfinder\stores\model\swin_large_patch4_window7_224_swin_large_patch4_window7_224_10_folds_3lcg145y"
# 1. Push all inferenced models and oof and submissions to the same folder with the model weights.


def inference_all_folds(
    model: models.CustomNeuralNet,
    state_dicts: List[collections.OrderedDict],
    test_loader: torch.utils.data.DataLoader,
) -> np.ndarray:
    """Inference the model on all K folds.

    Args:
        model (models.CustomNeuralNet): The model to be used for inference. Note that pretrained should be set to False.
        state_dicts (List[collections.OrderedDict]): The state dicts of the models. Generally, K Fold means K state dicts.
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set.

    Returns:
        mean_preds (np.ndarray): The mean of the predictions of all folds.
    """

    model.to(device)
    model.eval()

    with torch.no_grad():
        all_folds_probs = []

        for _fold_num, state in enumerate(state_dicts):
            if "model_state_dict" not in state:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state["model_state_dict"])

            current_fold_probs = []

            for data in tqdm(test_loader, position=0, leave=True):
                images = data["X"].to(device, non_blocking=True)
                test_logits = model(images)
                test_probs = (
                    trainer.Trainer.get_sigmoid_softmax()(test_logits)
                    .cpu()
                    .numpy()
                )
                test_probs = test_probs * 100  # only for petfinder!

                current_fold_probs.append(test_probs)

            current_fold_probs = np.concatenate(current_fold_probs, axis=0)
            all_folds_probs.append(current_fold_probs)
        mean_preds = np.mean(all_folds_probs, axis=0)
    return mean_preds


def inference(
    df_test: pd.DataFrame,
    model_dir: Union[str, Path],
    model: Union[models.CustomNeuralNet, Any],
    transform_dict: Dict[str, albumentations.Compose],
    df_sub: pd.DataFrame = None,
) -> Dict[str, np.ndarray]:

    """Inference the model and perform TTA, if any.

    Dataset and Dataloader are constructed within this function because of TTA.
    model and transform_dict are passed as arguments to enable inferencing multiple different models.

    Args:
        df_test (pd.DataFrame): The test dataframe.
        model_dir (str, Path): model directory for the model.
        model (Union[models.CustomNeuralNet, Any]): The model to be used for inference. Note that pretrained should be set to False.
        transform_dict (Dict[str, albumentations.Compose]): The dictionary of transforms to be used for inference. Should call from get_inference_transforms().
        df_sub (pd.DataFrame, optional): The submission dataframe. Defaults to None.

    Returns:
        all_preds (Dict[str, np.ndarray]): {"normal": normal_preds, "tta": tta_preds}
    """

    if df_sub is None:
        config.logger.info(
            "No submission dataframe detected, setting df_sub to be df_test."
        )
        df_sub = df_test.copy()

    all_preds = {}
    model = model.to(device)

    # Take note I always save my torch models as .pt files. Note we must return paths as str as torch.load does not support pathlib.
    weights = utils.return_list_of_files(
        directory=model_dir, return_string=True, extension=".pt"
    )

    state_dicts = [torch.load(path)["model_state_dict"] for path in weights]

    preds_list = []
    # Loop over each TTA transforms, if TTA is none, then loop once over normal inference_augs.
    for aug_name, aug_param in transform_dict.items():
        test_dataset = dataset.CustomDataset(
            df=df_test, transforms=aug_param, mode="test"
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **LOADER_PARAMS.test_loader
        )
        predictions = inference_all_folds(
            model=model, state_dicts=state_dicts, test_loader=test_loader
        )
        all_preds[aug_name] = predictions

        ################# To change when necessary depending on the metrics needed for submission #################
        # TODO: Consider returning a list of predictions ranging from np.argmax to preds, probs etc, and this way we can use whichever from the output? See my petfinder for more.
        df_sub[FOLDS.class_col_name] = predictions

        df_sub[[FOLDS.image_col_name, FOLDS.class_col_name]].to_csv(
            Path(
                MODEL_ARTIFACTS_PATH, f"submission_{aug_name}.csv", index=False
            )
        )
        print(df_sub.head())

        plt.figure(figsize=(12, 6))
        plt.hist(df_sub[FOLDS.class_col_name], bins=100)
    return all_preds


def compute_rmse_oof(model_dir: Union[str, Path]) -> np.ndarray:
    """Compute the RMSE of the oof predictions."""

    # Take note I always save my torch models as .pt files. Note we must return paths as str as torch.load does not support pathlib.
    weights = utils.return_list_of_files(
        directory=model_dir, return_string=True, extension=".pt"
    )
    oof_trues = [torch.load(path)["oof_trues"] for path in weights]
    oof_probs = [torch.load(path)["oof_probs"] for path in weights]

    oof_trues = torch.cat(oof_trues, dim=0) * 100
    oof_probs = torch.cat(oof_probs, dim=0) * 100

    oof_rmse = metrics.mse_torch(oof_trues, oof_probs)
    print(oof_rmse)
