from typing import Dict, Union

import numpy as np
import torch
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from config import global_params

TRANSFORMS = global_params.AugmentationParams()


def get_train_transforms(
    image_size: int = TRANSFORMS.image_size,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on training data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        (albumentations.core.composition.Compose): [description]
    """
    return albumentations.Compose(
        [
            albumentations.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333333333333333),
            ),
            albumentations.RandomRotate90(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Cutout(p=0.5),
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(
    image_size: int = TRANSFORMS.image_size,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on validation data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_gradcam_transforms(
    image_size: int = TRANSFORMS.image_size,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on gradcam data.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        [type]: [description]
    """
    return albumentations.Compose(
        [
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(
                mean=TRANSFORMS.mean,
                std=TRANSFORMS.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_inference_transforms(
    image_size: int = TRANSFORMS.image_size,
) -> Dict[str, albumentations.core.composition.Compose]:
    """Performs Augmentation on test dataset.
    Returns the transforms for inference in a dictionary which can hold TTA transforms.

    Args:
        image_size (int, optional): [description]. Defaults to TRANSFORMS.image_size.

    Returns:
        Dict[str, albumentations.Compose]: [description]
    """
    # TODO: Remember tta transforms need resize and normalize.
    transforms_dict = {
        "transforms_test": albumentations.Compose(
            [
                albumentations.Resize(image_size, image_size),
                albumentations.Normalize(
                    mean=TRANSFORMS.mean,
                    std=TRANSFORMS.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        ),
        # "tta_hflip": albumentations.Compose(
        #     [
        #         albumentations.HorizontalFlip(p=1.0),
        #         albumentations.Resize(image_size, image_size),
        #         ToTensorV2(),
        #     ]
        # ),
    }

    return transforms_dict


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    params: TRANSFORMS.mixup_params,
) -> torch.Tensor:
    """Implements mixup data augmentation.

    Args:
        x (torch.Tensor): The input tensor.
        y (torch.Tensor): The target tensor.
        params (TRANSFORMS, optional): [description]. Defaults to TRANSFORMS.mixup_params.

    Returns:
        torch.Tensor: [description]
    """

    # TODO: https://www.kaggle.com/reighns/petfinder-image-tabular check this to add z if there are dense targets.
    assert params["mixup_alpha"] > 0, "Mixup alpha must be greater than 0."
    assert (
        x.size(0) > 1
    ), "Mixup requires more than one sample as at least two samples are needed to mix."

    if params["mixup_alpha"] > 0:
        lambda_ = np.random.beta(params["mixup_alpha"], params["mixup_alpha"])
    else:
        lambda_ = 1

    batch_size = x.size()[0]
    if params["use_cuda"] and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lambda_


def mixup_criterion(
    criterion: Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss],
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Implements mixup criterion.

    Args:
        criterion (Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]): [description]
        logits (torch.Tensor): [description]
        y_a (torch.Tensor): [description]
        y_b (torch.Tensor): [description]
        lambda_ (float): [description]

    Returns:
        torch.Tensor: [description]
    """
    return lambda_ * criterion(logits, y_a) + (1 - lambda_) * criterion(
        logits, y_b
    )
