from __future__ import generators, print_function


import sys

from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


BASE_DIR = Path(__file__).parent.parent.absolute().__str__()
sys.path.append(BASE_DIR)


import pandas as pd

import torch

import typer
from sklearn import metrics

from src import (
    plot,
    prepare,
    transformation,
    utils,
    models,
    inference,
    trainer,
    dataset,
    lr_finder,
    metrics,
)
from config import config, global_params
from torch._C import device

import gc
import wandb
import matplotlib.pyplot as plt
import shutil

FILES = global_params.FilePaths()
FOLDS = global_params.MakeFolds()
MODEL_PARAMS = global_params.ModelParams()
LOADER_PARAMS = global_params.DataLoaderParams()
TRAIN_PARAMS = global_params.GlobalTrainParams()
WANDB_PARAMS = global_params.WandbParams()
LOGS_PARAMS = global_params.LogsParams()

device = config.DEVICE

main_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "main.log"),
    module_name="main",
)

shutil.copy(FILES.global_params_path, LOGS_PARAMS.LOGS_DIR_RUN_ID)

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Load data from URL and save to local drive."""
    # Download data, pre-caching.
    # datasets.MNIST(root=config.DATA_DIR.absolute(), train=True, download=True)
    # datasets.MNIST(root=config.DATA_DIR.absolute(), train=False, download=True)
    # Save data

    main_logger.info("Data downloaded!")


def wandb_init(fold: int):
    """Initialize wandb run.

    Args:
        fold (int): [description]

    Returns:
        [type]: [description]
    """
    config = {
        "Train_Params": TRAIN_PARAMS.to_dict(),
        "Model_Params": MODEL_PARAMS.to_dict(),
        "Loader_Params": LOADER_PARAMS.to_dict(),
        "File_Params": FILES.to_dict(),
        "Wandb_Params": WANDB_PARAMS.to_dict(),
        "Folds_Params": FOLDS.to_dict(),
        "Augment_Params": global_params.AugmentationParams().to_dict(),
        "Criterion_Params": global_params.CriterionParams().to_dict(),
        "Scheduler_Params": global_params.SchedulerParams().to_dict(),
        "Optimizer_Params": global_params.OptimizerParams().to_dict(),
    }

    wandb_run = wandb.init(
        config=config,
        name=f"{TRAIN_PARAMS.model_name}_fold_{fold}",
        **WANDB_PARAMS.to_dict(),
    )
    return wandb_run


def log_gradcam(curr_fold_best_checkpoint, df_oof, plot_gradcam: bool = True):
    """Log gradcam images into wandb for error analysis.
    # TODO: Consider getting the logits for error analysis, for example, if a predicted image which is correct has high logits this means the model is very sure, conversely, if a predicted image has low logits and also wrong, we also check why.
    """

    wandb_table = wandb.Table(
        columns=[
            "image_id",
            "y_true",
            "y_pred",
            "original_image",
            "gradcam_image",
        ]
    )
    model = models.CustomNeuralNet(pretrained=False)

    # I do not need to do the following as the trainer returns a checkpoint model.
    # So we do not need to say: model = CustomNeuralNet(pretrained=False) -> state = torch.load(...)
    curr_fold_best_state = curr_fold_best_checkpoint["model_state_dict"]
    model.load_state_dict(curr_fold_best_state)
    model.to(device)
    model.eval()

    if "vit" in MODEL_PARAMS.model_name:
        # blocks[-1].norm1  # for vit models use this, note this is using TIMM backbone.
        target_layers = [model.backbone.blocks[-1].norm1]
    elif "efficientnet" in MODEL_PARAMS.model_name:
        target_layers = [model.backbone.conv_head]
        reshape_transform = None
    elif "resnet" in MODEL_PARAMS.model_name:
        target_layers = [model.backbone.layer4[-1]]

    elif "swin" in MODEL_PARAMS.model_name:
        # https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/swinT_example.py
        def reshape_transform(tensor, height=7, width=7):
            # height, width 12 for swin 384
            result = tensor.reshape(
                tensor.size(0), height, width, tensor.size(2)
            )

            # Bring the channels to the first dimension,
            # like in CNNs.
            result = result.permute(0, 3, 1, 2)
            return result

        target_layers = [model.backbone.layers[-1].blocks[-1].norm1]

    # load gradcam_dataset
    gradcam_dataset = dataset.CustomDataset(
        df=df_oof,
        transforms=transformation.get_gradcam_transforms(),
        mode="gradcam",
    )
    count = 0
    for data in gradcam_dataset:
        X, y, original_image, image_id = (
            data["X"],
            data["y"],
            data["original_image"],
            data["image_id"],
        )
        X_unsqueezed = X.unsqueeze(0)
        gradcam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=device,
            reshape_transform=reshape_transform,
        )

        gradcam_output = gradcam(input_tensor=X_unsqueezed)
        original_image = original_image.cpu().detach().numpy() / 255.0
        y_true = y.cpu().detach().numpy()
        y_pred = df_oof.loc[
            df_oof[FOLDS.image_col_name] == image_id, "oof_preds"
        ].values[0]

        assert (
            original_image.shape[-1] == 3
        ), "Channel Last when passing into gradcam."

        gradcam_image = show_cam_on_image(
            original_image, gradcam_output[0], use_rgb=False
        )
        if plot_gradcam:
            _fig, axes = plt.subplots(figsize=(8, 8), ncols=2)
            axes[0].imshow(original_image)
            axes[0].set_title(f"y_true={y_true:.4f}")
            axes[1].imshow(gradcam_image)
            axes[1].set_title(f"y_pred={y_pred}")
            plt.show()
            torch.cuda.empty_cache()

        # No idea why we must cast to float instead of just numpy.
        wandb_table.add_data(
            image_id,
            float(y_true),
            float(y_pred),
            wandb.Image(original_image),
            wandb.Image(gradcam_image),
        )
        # TODO: take 10 correct predictions and 10 incorrect predictions.
        # TODO: needs modification if problem is say regression, or multilabel.
        count += 1
        if count == 10:
            break
    return wandb_table


def train_one_fold(
    df_folds: pd.DataFrame,
    fold: int,
    is_plot: bool = False,
    is_forward_pass: bool = True,
    is_gradcam: bool = True,
    is_find_lr: bool = False,
):
    """Train the model on the given fold."""

    ################################## W&B #####################################
    # wandb.login()
    wandb_run = wandb_init(fold=fold)

    train_loader, valid_loader, df_oof = prepare.prepare_loaders(df_folds, fold)

    if is_plot:
        _image_grid = plot.show_image(
            loader=train_loader,
            nrows=1,
            ncols=1,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        # TODO: add image_grid to wandb.

    # Model, cost function and optimizer instancing
    model = models.CustomNeuralNet().to(device)

    if is_forward_pass:
        # Forward Sanity Check
        # TODO: https://discuss.pytorch.org/t/why-cannot-i-call-dataloader-or-model-object-twice/137761
        # Find out why this will change model behaviour, use with caution, or maybe just put it outside this function for safety.
        _forward_X, _forward_y = models.forward_pass(
            loader=train_loader, model=model
        )
    if is_find_lr:
        lr_finder.find_lr(
            model, device, train_loader, valid_loader, use_valid=False
        )

    reighns_trainer: trainer.Trainer = trainer.Trainer(
        params=TRAIN_PARAMS,
        model=model,
        device=device,
        wandb_run=wandb_run,
    )

    curr_fold_best_checkpoint = reighns_trainer.fit(
        train_loader, valid_loader, fold
    )

    # TODO: Note that for sigmoid on one class, the OOF score is the positive class.
    df_oof[[f"class_{str(c)}_oof" for c in range(TRAIN_PARAMS.num_classes)]] = (
        curr_fold_best_checkpoint["oof_probs"].detach().numpy()
    )

    df_oof["oof_trues"] = curr_fold_best_checkpoint["oof_trues"]
    df_oof["oof_preds"] = curr_fold_best_checkpoint["oof_preds"]

    df_oof.to_csv(Path(FILES.oof_csv, f"oof_fold_{fold}.csv"), index=False)
    if is_gradcam:
        # TODO: df_oof['error_analysis'] = todo - error analysis by ranking prediction confidence and plot gradcam for top 10 and bottom 10.
        gradcam_table = log_gradcam(
            curr_fold_best_checkpoint=curr_fold_best_checkpoint,
            df_oof=df_oof,
            plot_gradcam=False,
        )

        wandb_run.log({"gradcam_table": gradcam_table})
        utils.free_gpu_memory(gradcam_table)

    utils.free_gpu_memory(model)
    wandb_run.finish()  # Finish the run to start next fold.

    return df_oof


def train_loop(*args, **kwargs):
    """Perform the training loop on all folds. Here The CV score is the average of the validation fold metric.
    While the OOF score is the aggregation of all validation folds."""

    df_oof = pd.DataFrame()

    for fold in range(1, FOLDS.num_folds + 1):
        _df_oof = train_one_fold(*args, fold=fold, **kwargs)
        df_oof = pd.concat([df_oof, _df_oof])

        # TODO: populate the cv_score_list using a dataframe like breast cancer project.
        # curr_fold_best_score_dict, curr_fold_best_score = get_oof_roc(config, _oof_df)
        # cv_score_list.append(curr_fold_best_score)
        # print("\n\n\nOOF Score for Fold {}: {}\n\n\n".format(fold, curr_fold_best_score))

    # cv_mean_d, cv_std_d = metrics.calculate_cv_metrics(df_oof)
    # main_logger.info(f"\n\n\nMEAN CV: {cv_mean_d}\n\n\nSTD CV: {cv_std_d}")

    # print("Five Folds OOF", get_oof_roc(config, oof_df))

    df_oof.to_csv(Path(FILES.oof_csv, "oof.csv"), index=False)

    return df_oof


if __name__ == "__main__":
    utils.seed_all()

    # @Step 1: Download and load data.
    df_train, df_test, df_folds, df_sub = prepare.prepare_data()

    is_inference = False
    if not is_inference:
        df_oof = train_loop(
            df_folds=df_folds, is_plot=False, is_forward_pass=True
        )

    # model_dir = Path(FILES.weight_path, MODEL_PARAMS.model_name).__str__()
    else:
        model_dir = r"C:\Users\reighns\kaggle_projects\cassava\model\tf_efficientnet_b0_ns"
        predictions = inference.inference(df_test, model_dir, df_sub)
        # _ = inference.show_gradcam()
