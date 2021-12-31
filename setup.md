## Specific Changes

### Sturge's Rule

The change can be found in `make_folds` and `MakeFolds` params and `prepare`'s `prepare_data` function. I did not make this change to the main PyTorch Pipeline as it might cause clutter and confusion. 

### Normalize Pawpularity

Normalize to 0 to 1 range for the target Pawpularity.

### Get Regression Metrics

Added RMSE metrics in `metrics`.

```python
def get_regression_metrics(
    self,
    y_trues: torch.Tensor,
    y_preds: torch.Tensor,
    y_probs: torch.Tensor,
):
    ### ONLY FOR THIS COMP YOU NEED TO DENORMALIZE ###
    y_trues = y_trues * 100
    y_probs = y_probs * 100
    mse = mse_torch(y_trues, y_probs, is_rmse=False)
    rmse = mse_torch(y_trues, y_probs, is_rmse=True)
    
    return {"mse": mse, "rmse": rmse}
```
Consider adding to main pipeline..