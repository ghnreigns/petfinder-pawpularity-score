- https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301015
  - We all know that normal `resize` an image of dimensions say `600x800` to `256x256` may result in the image squishing, and losing information in pixels. A better approach is `RandomResizedCrop` which randomly selects a region of the image and resizes it to `256x256` without any loss of information. 
  - **FASTAI**: https://docs.fast.ai/vision.augment.html#Resize resize tricks. Seems like the top people use this trick even if not using fastai, see CPMP and Chris.
    - A.SmallestMaxSize(max_size=dim, p=1.0), A.RandomCrop(height=dim, width=dim, p=1.0) for train
    - A.SmallestMaxSize(max_size=dim, p=1.0), A.CenterCrop(height=dim, width=dim, p=1.0) for validation
- Mixup: Put mixup into the mix~
- Divide Norm Layers: `models.divide_norm_layers` and `trainer.get_divide_norm_optimizer`
- How to freeze layers for effnet: https://www.kaggle.com/max237/resnet50-transfer-learning-in-pytorch
- Do TTA last as I am experienced in it, but rmb fastai's beta is a weight for tta blend, for me my beta is akin to tta say n times and inference without tta and divide by n+1.
- Embeddings
- https://www.kaggle.com/ytakayama/train-pytorch-swin-5fold-some-tips and https://www.kaggle.com/ytakayama/inference-pytorch-swin-5fold-notta seems good, even chris follows.
- Use Sturge


## Additional

- models.freeze_batchnorm_layers
- models.get_conv_layers
- models.divide_norm_layers
- trainer.get_divide_norm_optimizer
- make_folds.sturge
- Added doc string for mixup: also do up a notebook for mixup to visualize and work out the math.
- Add model params in ModelParams. Do in Cassava too.
- talk bout lr finder?

## Experiment 1

### Problem Statement

- Transform Regression to Classification. Regression problems can be treated as classification assuming there is a lower and upper bound on the labels and they are evenly spaced
- 

### Cross Validation

- Note that we are transforming a regression problem to a classification problem. So we have `is_normalize` flag to indicate whether we are doing regression or classification by dividing the target by the max value.
- Use [Sturge's Rule](https://www.vedantu.com/question-answer/the-sturges-rule-for-determining-the-number-of-class-11-maths-cbse-5fb494704ad6c23c32da71f2) to get the number of classes. i.e. we have an artifical 100 classes, then it is quite difficult to just use StratifiedKFold on 100 classes with the given data. Instead, we use Sturge's Rule to bin the 100 classes to a suitable number of classes. i.e class 1-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71-80, 81-90, 91-100 -> 10 classes.
- StratifiedKFold is then used on the "new target" which we obtained from Sturge's Rule.


### Dataset and DataLoader

#### Dataset

Nothing fancy happening here, something worth noting is that I did the `flatten()` operation whenever the criterion/loss function is `BCEWithLogitsLoss` as PyTorch expects the `target` to be of the shape `(batch_size, 1)` and not `(batch_size,)`. So if I flatten here, then `DataLoader` will automatically collate the flattened `targets` from `(batch_size,)` to `(batch_size, 1)`.

I also returned the `original_image` resized to the training image size so that I can use it for `gradcam`.

#### DataLoader Params

With GPU limitations, the `batch_size` is set to `8`. As usual, during training, we set `shuffle=True` but during validation, we set `shuffle=False` as we want to keep the order of the images. 

```python
@dataclass
class DataLoaderParams:
    """Class to keep track of the data loader parameters."""

    train_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
            "shuffle": True,
            "collate_fn": None,
        }
    )
    valid_loader: Dict[str, Any] = field(
        default_factory=lambda: {
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
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
```

---

### Augmentations

#### Resize Methodology

`RandomResizedCrop` is a good choice for the the training regime as it randomly selects a region of the image and resizes it to `image_size x image_size`. This is better than `Resize` as it allows the model to generalize better. One can imagine that if the model is trained on a small image size, but in each epoch it sees a different crop/part of the image, then effectively we are training on a larger image size.

In this competition, Fastai seems to perform well, to mimic their regime, we can use the following as detailed in the [fastai docs](https://docs.fast.ai/vision.augment.html#Resize):

```python
# Training
albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=image_size, p=1.0),
            albumentations.RandomCrop(
                height=image_size, width=image_size, p=1.0
            ),
        ]
    ) 

# Validation
albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=image_size, p=1.0),
            albumentations.CenterCrop(
                height=image_size, width=image_size, p=1.0
            ),
        ]
    ) 

# Inference
albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=image_size, p=1.0),
            albumentations.CenterCrop(
                height=image_size, width=image_size, p=1.0
            ),
        ]
    ) 
```

We first **rescale (not resize)** using `SmallestMaxSize` to maintain the aspect ratio of the image. Then we **crop** using `RandomCrop` to ensure that the crop is of the same size as the image. Note that if you use `swin_transformer` then one must conform to the image size as the model does not allow dynamic image size (unless you do progressive resize).

For a visual idea of why the above regimen is better than a simpel `Resize`, check out Chris's [discussion](https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301015).

> **Important Note:** If you use the above regime during training, then during inference, do not forget to use back the same resize methodology. 

#### Normalization

Imagenet weights are used for the training regime. One can however, experiment with the image set's own native mean and std. For example, if the training set is a subset of the imagenet training set (or very closely resemble images inside Imagenet), then one can use the imagenet mean and std for training. However, if the training set has a distribution shift from the imagenet distribution, such as very complicated medical images, then one may consider calculating the mean and std from the training set at hand.

```python
albumentations.Normalize(
    mean=TRANSFORMS.mean,
    std=TRANSFORMS.std,
    max_pixel_value=255.0,
    p=1.0,
)
```

#### Light vs Heavy Augs

I noticed many experienced practitioners use `Light` augmentation for lower image size and `Heavy` augmentation for higher image size. I will check and understand why.

#### Mixup

A strong regularization technique for training. Consider image_1 and image_2 are two images from the same class. Then, we can randomly select a point in the image_1 and image_2 and swap the pixel values at that point. In laymen, you get a cat and dog image, then the "mixed" image has both hints of cat and dog.

KERAS has a good [documentation](https://keras.io/examples/vision/mixup/). [Original Paper](https://arxiv.org/abs/1710.09412)

Here I set the hyperparameters to be `alpha=0.3`.

```python
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
        criterion (Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]): The loss function.
        logits (torch.Tensor): The logits tensor.
        y_a (torch.Tensor): The target of image a.
        y_b (torch.Tensor): The target of image b.
        lambda_ (float): The lambda value from mixup.

    Returns:
        torch.Tensor: The mixed criterion (loss).
    """
    return lambda_ * criterion(logits, y_a) + (1 - lambda_) * criterion(
        logits, y_b
    )
```

---

### Loss

We used the [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss) for the training regime. This is a classification loss but our task is actually regression. Our regression has values from 0 to 100, if we normalize the regression values to be between 0 and 1, then we can use the [BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.BCEWithLogitsLoss) as our loss.

One might wonder that the normalized targets will still be continuous on `[0, 1]` and not a binary 0 and 1. So will the `BCEWithLogitsLoss` still work as it intended to? Then please read the following posts. 

- https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/275461
- https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/284040
- https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/275094

> **Important PSA:** The `BCEWithLogitsLoss` combines the `sigmoid` and `BCELoss` in one loss function. Thus, I passed in raw logits into the loss function and not the sigmoid of the logits. According to docs, this is numerically more stable.


---

### Model Architecture

```python
  model_name: str = "swin_large_patch4_window7_224"  
  pretrained: bool = True
  input_channels: int = 3
  output_dimension: int = 1
```

#### Backbone

Favourite model in the competition goes to `swin_large_patch4_window7_224`. If you feel adventurous, you can read a KERAS implementation of this model [here](https://keras.io/examples/vision/swin_transformers/) and the original paper [here](https://arxiv.org/abs/2103.14030).

The general idea is that we are not merely predicting whether the image is a cat or a dog, but also the surrounding area. From both the original paper, and empically from `gradcam`, the transformers focus on "global attention" and thus the surrounding area is taken into account. This is one hypothesis that the transformer models are outperforming the CNN models. After all, if you have a kinda cute dog photo, but taken in a dirty bathroom (surrounding), then the overall cuteness score will be lower.


#### Head

I used an `OrderedDict` so I can easily access the layers by name. This is useful for me to use the image embeddings.

The following architecture is based on one of the public notebooks from Kaggle. Surprisingly, no intermediate activations were used.

In my next experiment, I will try to put in a few intermediate activations such as `GeLu` and `Swish` to see if the model performs better.


```python
self.single_head_fc = torch.nn.Sequential(
    OrderedDict(
        [
            ("linear_1", torch.nn.Linear(self.in_features, 128)),
            ("dropout_1", torch.nn.Dropout(p=0.1)),
            ("linear_2", torch.nn.Linear(128, 64)),
            ("linear_3", torch.nn.Linear(64, self.out_features)),
        ]
    )
)
```

### Optimizer and Divide Bias

We use `AdamW` optimizer but using Fastai trick. We also used this [notebook](https://www.kaggle.com/ytakayama/train-pytorch-swin-5fold-some-tips/notebook) to mimic Fastai's training regime.

With an easy to understand example from PyTorch's [optim](https://pytorch.org/docs/stable/optim.html):

```python
optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
```

means that model.base's parameters will use the default learning rate of 1e-2, model.classifier's parameters will use a learning rate of 1e-3, and a momentum of 0.9 will be used for all parameters.

We use a similar idea here but pertaining to Batch Norm Layers.

```python
def get_divide_norm_optimizer(self):
    """Get the optimizer for the model using fastai method.

    Returns:
        [type]: [description]
    """
    norm_bias_params, non_norm_bias_params = models.divide_norm_bias(
        self.model
    )
    opt_wd_non_norm_bias = 0.01
    opt_wd_norm_bias = 0
    opt_beta1 = 0.9
    opt_beta2 = 0.99
    opt_eps = 1e-5

    optimizer = torch.optim.AdamW(
        [
            {
                "params": norm_bias_params,
                "weight_decay": opt_wd_norm_bias,
            },
            {
                "params": non_norm_bias_params,
                "weight_decay": opt_wd_non_norm_bias,
            },
        ],
        betas=(opt_beta1, opt_beta2),
        eps=opt_eps,
        lr=6e-6,
        amsgrad=False,
    )
    return optimizer
```


#### Pitfalls of Dropout and Batch Normalization

Read Kaggle GM post on why dropout and freezing BN layers are recommended in regression.

- https://www.kaggle.com/c/commonlitreadabilityprize/discussion/260729#1442448
- https://towardsdatascience.com/pitfalls-with-dropout-and-batchnorm-in-regression-problems-39e02ce08e4d


### Learning Rate Scheduler

We use fastai's LR Finder to find an optimal learning rate.



