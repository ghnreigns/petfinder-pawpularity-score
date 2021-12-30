## GPU Memory Usage

Delete model, optimizer, and scheduler to free up memory. For example, when you do a forward pass, you created the model, please delete it after the forward pass. Also when using LR finder, you need to create the model, optimizer and scheduler. Please delete them.

Sequence of events, This order of operations does the trick for me, removing the parameters and gradients from the GPU.

```python
delete model, optimizer and scheduler
gc.collect()
torch.cuda.empty_cache()
```

Source: https://github.com/huggingface/transformers/issues/1742
Source: https://discuss.pytorch.org/t/memory-leak-debugging-and-common-causes/67339 On how to debug using utils.show_gpu_usage()

## Weights & Biases

```bash
# CLI Interface
pip install wandb --upgrade # upgrade
wandb login --relogin # Login to wandb
# Subsequently, you can run your wandb inside your script.
# TODO: To explore other options like put the wandb keys in a config file.
```


## Prepare Data

```python
df_train, df_test, df_folds, sub = prepare.prepare_data()
```
Returns the dataframes for the training, testing, and folds data, and the submission dataframe.

[]: # Language: python
[]: # Path: prepare.py


## Plot Dataloader

Initialize the dataloader and plot random images.

```python
# Plot random images from dataloader for sanity check.
train_dataset = dataset.CustomDataset(
    df=df_folds,
    transforms=transformation.get_train_transforms(),
    mode="train",
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    # sampler=RandomSampler(dataset_train),
    **loader_params.train_loader,
    worker_init_fn=utils.seed_worker,
)

plot.show_image(
    loader=train_loader,
    nrows=1,
    ncols=1,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
```

## Forward Pass

As a sanity check of our model, we can perform a forward pass.

```python
forward_X, forward_y, model_summary = models.forward_pass(
    model=models.CustomNeuralNet()
)
```

## Augmentations

[Reference on RandomResizedCrop](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/). So albumentation's has default parameters same as the article.