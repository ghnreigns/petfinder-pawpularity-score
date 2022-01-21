- https://www.kaggle.com/c/petfinder-pawpularity-score/discussion/301015
  - We all know that normal `resize` an image of dimensions say `600x800` to `256x256` may result in the image squishing, and losing information in pixels. A better approach is `RandomResizedCrop` which randomly selects a region of the image and resizes it to `256x256` without any loss of information. 

- Mixup: hasnt try!

## Additional

- models.freeze_batchnorm_layers
- models.get_conv_layers