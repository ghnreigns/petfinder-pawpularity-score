import collections
import glob
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import config, global_params
from tqdm.auto import tqdm

from src import dataset, models, transformation, trainer

MODEL = global_params.ModelParams()
FOLDS = global_params.MakeFolds()
LOADER_PARAMS = global_params.DataLoaderParams()
device = config.DEVICE


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
        all_folds_preds = []

        for _fold_num, state in enumerate(state_dicts):
            if "model_state_dict" not in state:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state["model_state_dict"])

            current_fold_preds = []

            for data in tqdm(test_loader, position=0, leave=True):
                images = data["X"].to(device, non_blocking=True)
                logits = model(images)
                test_prob = (
                    trainer.Trainer.get_sigmoid_softmax()(logits).cpu().numpy()
                )

                current_fold_preds.append(test_prob)

            current_fold_preds = np.concatenate(current_fold_preds, axis=0)
            all_folds_preds.append(current_fold_preds)
        mean_preds = np.mean(all_folds_preds, axis=0)
    return mean_preds


# TODO: See my latest PyTorch to change the transform outside of function and as an argument.
# TODO: Move model as argument too.


def inference(
    df_test: pd.DataFrame,
    model_dir: str,
    df_sub: pd.DataFrame = None,
) -> Dict[str, np.ndarray]:
    """Inference the model and perform TTA, if any.

    Dataset and Dataloader are constructed within this function because of TTA.

    Args:
        df_test (pd.DataFrame): The test dataframe.
        model_dir (str): model directory for the model.
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

    model = models.CustomNeuralNet(pretrained=False).to(device)

    transform_dict = transformation.get_inference_transforms()

    # TODO: glob.glob does not preserve sequence... means we need order by lexiographic order. sorted(list([model_path for model_path in glob.glob(model_dir + "/*.pt")]))
    weights = [model_path for model_path in glob.glob(model_dir + "/*.pt")]
    state_dicts = [torch.load(path)["model_state_dict"] for path in weights]

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
        df_sub[FOLDS.class_col_name] = np.argmax(predictions, axis=1)

        df_sub[[FOLDS.image_col_name, FOLDS.class_col_name]].to_csv(
            f"submission_{aug_name}.csv", index=False
        )
        print(df_sub.head())

        plt.figure(figsize=(12, 6))
        plt.hist(df_sub[FOLDS.class_col_name], bins=100)
    return all_preds
