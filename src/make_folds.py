import pandas as pd
from config import config, global_params
import numpy as np

# from IPython.display import display
from sklearn.model_selection import GroupKFold, StratifiedKFold

##################### CUSTOM Sturges' rule ####################################


def sturges_rule(df_train: pd.DataFrame) -> pd.DataFrame:
    """Sturge's rule for number of bins

    Args:
        df_train (pd.DataFrame): The train dataframe.

    Returns:
        df_train (pd.DataFrame): The train dataframe with an additional column "bin".
    """

    num_bins = int(np.floor(1 + np.log2(len(df_train))))
    # Cut the target Pawpularity into `num_bins` using pd.cut
    df_train["bins"] = pd.cut(
        df_train[global_params.MakeFolds().class_col_name],
        bins=num_bins,
        labels=False,
        ordered=True,
    )
    df_train["bins"].hist()
    return df_train


def make_folds(
    train_csv: pd.DataFrame, cv_params: global_params.MakeFolds()
) -> pd.DataFrame:
    """Split the given dataframe into training folds."""
    if cv_params.use_sturge is True:
        cv_params.class_col_name = "bins"

    if cv_params.cv_schema == "StratifiedKFold":
        df_folds = train_csv.copy()
        skf = StratifiedKFold(
            n_splits=cv_params.num_folds,
            shuffle=True,
            random_state=cv_params.seed,
        )

        for fold, (_train_idx, val_idx) in enumerate(
            skf.split(
                X=df_folds[cv_params.image_col_name],
                y=df_folds[cv_params.class_col_name],
            )
        ):
            df_folds.loc[val_idx, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)
        print(df_folds.groupby(["fold", cv_params.class_col_name]).size())

    elif cv_params.cv_schema == "GroupKfold":
        df_folds = train_csv.copy()
        gkf = GroupKFold(n_splits=cv_params.num_folds)
        groups = df_folds[cv_params.group_kfold_split].values
        for fold, (_train_index, val_index) in enumerate(
            gkf.split(
                X=df_folds, y=df_folds[cv_params.class_col_name], groups=groups
            )
        ):
            df_folds.loc[val_index, "fold"] = int(fold + 1)
        df_folds["fold"] = df_folds["fold"].astype(int)

        print(df_folds.groupby(["fold", cv_params.class_col_name]).size())

    else:
        config.logger.error(
            f"Unknown CV schema: {cv_params.cv_schema}, are you using custom split?"
        )
        df_folds = train_csv.copy()
        print(df_folds.groupby(["fold", cv_params.class_col_name]).size())

    df_folds.to_csv(cv_params.folds_csv, index=False)

    return df_folds
