import time
from pathlib import Path
from typing import DefaultDict, Dict, Union

import numpy as np
import torch
from config import config, global_params
import shutil

from tqdm.auto import tqdm
import os
from src import callbacks, metrics, models, utils, transformation

FILES = global_params.FilePaths()
CRITERION_PARAMS = global_params.CriterionParams()
SCHEDULER_PARAMS = global_params.SchedulerParams()
OPTIMIZER_PARAMS = global_params.OptimizerParams()
TRANSFORMS = global_params.AugmentationParams()
LOGS_PARAMS = global_params.LogsParams()

training_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "training.log"),
    module_name="training",
)
# TODO: Make use of gc.collect and torch.cuda.empty_cache to free up memory especially for transformers https://github.com/huggingface/transformers/issues/1742
# TODO: Consider saving image embeddings and everything under the sun in trainer, so we don't need to do it again during inference or PP.


class Trainer:
    """Object used to facilitate training."""

    def __init__(
        self,
        params: global_params.GlobalTrainParams,
        model: models.CustomNeuralNet,
        device=torch.device("cpu"),
        wandb_run=None,
        early_stopping: callbacks.EarlyStopping = None,
    ):
        # Set params
        self.params = params
        self.model = model
        self.device = device

        self.wandb_run = wandb_run

        self.early_stopping = early_stopping

        # TODO: To ask Ian if initializing the optimizer in constructor is a good idea? Should we init it outside of the class like most people do? In particular, the memory usage.
        if params.divide_norm_bias:
            self.optimizer = self.get_divide_norm_optimizer()
        else:
            self.optimizer = self.get_optimizer(
                model=self.model, optimizer_params=OPTIMIZER_PARAMS
            )
        self.scheduler = self.get_scheduler(
            optimizer=self.optimizer, scheduler_params=SCHEDULER_PARAMS
        )

        if self.params.use_amp:
            # https://pytorch.org/docs/stable/notes/amp_examples.html
            self.scaler = torch.cuda.amp.GradScaler()

        # list to contain various train metrics
        # TODO: how to add more metrics? wandb log too. Maybe save to model artifacts?
        self.monitored_metric = {
            "metric_name": "valid_rmse",
            "metric_score": None,
            "mode": "min",
        }
        # Metric to optimize, either min or max.
        self.best_valid_score = (
            -np.inf if self.monitored_metric["mode"] == "max" else np.inf
        )
        self.patience_counter = self.params.patience  # Early Stopping Counter
        self.history = DefaultDict(list)

        ########################################## Define Weight Paths ###################################################################
        # Note that now the model save path has wandb_run's group id appended for me to easily recover which run corresponds to which model.
        self.model_path = Path(
            FILES.weight_path,
            f"{self.params.model_name}_{self.wandb_run.group}",
        )
        # create model directory if not exist and model_directory with run_id to identify easily.
        Path.mkdir(self.model_path, exist_ok=True)

        ########################################################################################################################################

    @staticmethod
    def get_optimizer(
        model: models.CustomNeuralNet,
        optimizer_params: global_params.OptimizerParams = OPTIMIZER_PARAMS,
    ):
        """Get the optimizer for the model.

        Caution: Do not invoke self.model directly in this call as it may affect model initalization.
        READ: https://stackoverflow.com/questions/70107044/can-i-define-a-method-as-an-attribute

        Args:
            model (models.CustomNeuralNet): [description]
            optimizer_params (global_params.OptimizerParams): [description]

        Returns:
            [type]: [description]
        """

        return getattr(torch.optim, optimizer_params.optimizer_name)(
            model.parameters(), **optimizer_params.optimizer_params
        )

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

    @staticmethod
    def get_scheduler(
        optimizer: torch.optim,
        scheduler_params: global_params.SchedulerParams = SCHEDULER_PARAMS,
    ):
        """Get the scheduler for the optimizer.

        Args:
            optimizer (torch.optim): [description]
            scheduler_params (global_params.SchedulerParams(), optional): [description]. Defaults to SCHEDULER_PARAMS.scheduler_params.

        Returns:
            [type]: [description]
        """

        return getattr(
            torch.optim.lr_scheduler, scheduler_params.scheduler_name
        )(optimizer=optimizer, **scheduler_params.scheduler_params)

    # TODO: Write the type hints properly!
    @staticmethod
    def train_criterion(
        y_true: torch.Tensor,
        y_logits: torch.Tensor,
        batch_size: int,
        criterion_params: global_params.CriterionParams = CRITERION_PARAMS,
    ):
        """Train Loss Function.
        Note that we can evaluate train and validation fold with different loss functions.

        The below example applies for CrossEntropyLoss.

        Args:
            y_true ([type]): Input - N,C) where N = number of samples and C = number of classes.
            y_logits ([type]): If containing class indices, shape (N) where each value is 0 \leq \text{targets}[i] \leq C-10≤targets[i]≤C−1
                               If containing class probabilities, same shape as the input.
            criterion_params (global_params.CriterionParams, optional): [description]. Defaults to CRITERION_PARAMS.criterion_params.

        Returns:
            [type]: [description]
        """

        loss_fn = getattr(torch.nn, criterion_params.train_criterion_name)(
            **criterion_params.train_criterion_params
        )

        if criterion_params.train_criterion_name == "CrossEntropyLoss":
            pass
        elif criterion_params.train_criterion_name == "BCEWithLogitsLoss":
            assert (
                y_logits.shape[0] == y_true.shape[0] == batch_size
            ), f"BCEWithLogitsLoss expects first dimension to be batch size {batch_size}"
            assert (
                y_logits.shape == y_true.shape
            ), "BCEWithLogitsLoss inputs must be of the same shape."

        loss = loss_fn(y_logits, y_true)
        return loss

    @staticmethod
    def valid_criterion(
        y_true: torch.Tensor,
        y_logits: torch.Tensor,
        criterion_params: global_params.CriterionParams = CRITERION_PARAMS,
    ):
        """Validation Loss Function.

        Args:
            y_true ([type]): [description]
            y_logits ([type]): [description]
            criterion_params (global_params.CriterionParams, optional): [description]. Defaults to CRITERION_PARAMS.criterion_params.

        Returns:
            [type]: [description]
        """
        loss_fn = getattr(torch.nn, criterion_params.valid_criterion_name)(
            **criterion_params.valid_criterion_params
        )
        loss = loss_fn(y_logits, y_true)
        return loss

    @staticmethod
    def get_sigmoid_softmax() -> Union[torch.nn.Sigmoid, torch.nn.Softmax]:
        """Get the sigmoid or softmax function.

        Returns:
            Union[torch.nn.Sigmoid, torch.nn.Softmax]: [description]
        """
        if CRITERION_PARAMS.train_criterion_name == "BCEWithLogitsLoss":
            return getattr(torch.nn, "Sigmoid")()

        if CRITERION_PARAMS.train_criterion_name == "CrossEntropyLoss":
            return getattr(torch.nn, "Softmax")(dim=1)

    def get_classification_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        """[summary]

        Args:
            y_trues (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1); (May be float if using BCEWithLogitsLoss)
            y_preds (torch.Tensor): dtype=[torch.int64], shape=(num_samples, 1);
            y_probs (torch.Tensor): dtype=[torch.float32], shape=(num_samples, num_classes);

        Returns:
            [type]: [description]
        """
        # TODO: To implement Ian's Results class here so that we can return as per the following link: https://ghnreigns.github.io/reighns-ml-website/supervised_learning/classification/breast_cancer_wisconsin/Stage%206%20-%20Modelling%20%28Preprocessing%20and%20Spot%20Checking%29/
        # TODO: To think whether include num_classes, threshold etc in the arguments.
        torchmetrics_accuracy = metrics.accuracy_score_torch(
            y_trues,
            y_preds,
            num_classes=self.params.num_classes,
            threshold=0.5,
        )

        auroc_dict = metrics.multiclass_roc_auc_score_torch(
            y_trues,
            y_probs,
            num_classes=self.params.num_classes,
        )

        _auroc_all_classes, macro_auc = (
            auroc_dict["auroc_per_class"],
            auroc_dict["macro_auc"],
        )

        # TODO: To check robustness of the code for confusion matrix.
        # macro_cm = metrics.tp_fp_tn_fn_binary(
        #     y_true=y_trues, y_prob=y_probs, class_labels=[0, 1, 2, 3, 4]
        # )

        return {"accuracy": torchmetrics_accuracy, "macro_auroc": macro_auc}

    def get_regression_metrics(
        self,
        y_trues: torch.Tensor,
        y_preds: torch.Tensor,
        y_probs: torch.Tensor,
    ):
        ### ONLY FOR THIS COMP YOU NEED TO DENORMALIZE ###
        y_trues = y_trues * 100
        y_probs = y_probs * 100
        mse = metrics.mse_torch(y_trues, y_probs, is_rmse=False)
        rmse = metrics.mse_torch(y_trues, y_probs, is_rmse=True)

        return {"mse": mse, "rmse": rmse}

    @staticmethod
    def get_lr(optimizer: torch.optim) -> float:
        """Get the learning rate of the current epoch.
        Note learning rate can be different for different layers, hence the for loop.
        Args:
            self.optimizer (torch.optim): [description]
        Returns:
            float: [description]
        """
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        fold: int = None,
    ):
        """[summary]

        Args:
            train_loader (torch.utils.data.DataLoader): [description]
            val_loader (torch.utils.data.DataLoader): [description]
            fold (int, optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """

        # To automatically log gradients
        self.wandb_run.watch(self.model, log_freq=100)
        self.best_valid_loss = np.inf

        training_logger.info(
            f"\nTraining on Fold {fold} and using {self.params.model_name}\n"
        )

        for _epoch in range(1, self.params.epochs + 1):

            # get current epoch's learning rate
            curr_lr = self.get_lr(self.optimizer)

            ############################ Start of Training #############################

            train_start_time = time.time()

            train_one_epoch_dict = self.train_one_epoch(train_loader)
            train_loss = train_one_epoch_dict["train_loss"]

            # total time elapsed for this epoch
            train_time_elapsed = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - train_start_time)
            )

            training_logger.info(
                f"\n[RESULT]: Train. Epoch {_epoch}:\nAvg Train Summary Loss: {train_loss:.3f}\
                \nLearning Rate: {curr_lr:.5f}\nTime Elapsed: {train_time_elapsed}\n"
            )

            ########################### End of Training #################################

            ########################### Start of Validation #############################

            val_start_time = time.time()  # start time for validation
            valid_one_epoch_dict = self.valid_one_epoch(valid_loader)

            (
                valid_loss,
                valid_trues,
                valid_logits,
                valid_preds,
                valid_probs,
            ) = (
                valid_one_epoch_dict["valid_loss"],
                valid_one_epoch_dict["valid_trues"],
                valid_one_epoch_dict["valid_logits"],
                valid_one_epoch_dict["valid_preds"],
                valid_one_epoch_dict["valid_probs"],
            )

            # total time elapsed for this epoch
            valid_elapsed_time = time.strftime(
                "%H:%M:%S", time.gmtime(time.time() - val_start_time)
            )

            valid_metrics_dict = self.get_regression_metrics(
                valid_trues,
                valid_preds,
                valid_probs,
            )
            valid_rmse = valid_metrics_dict["rmse"]

            # TODO: Still need save each metric for each epoch into a list history. Rename properly
            # TODO: Log each metric to wandb and log file.

            training_logger.info(
                f"[RESULT]: Validation. Epoch {_epoch} | Avg Val Summary Loss: {valid_loss:.3f} | "
                f"Valid RMSE: {valid_rmse:.3f} | "
                f"Time Elapsed: {valid_elapsed_time}\n"
            )

            ########################### End of Validation ##############################

            ########################### Start of Wandb #################################
            self.history["epoch"].append(_epoch)
            self.history["valid_loss"].append(valid_loss)
            self.history["valid_rmse"].append(valid_rmse)
            self.log_metrics(_epoch, self.history)
            ########################### End of Wandb ###################################

            ########################## Start of Early Stopping ##########################
            ########################## Start of Model Saving ############################
            # TODO: Consider condensing early stopping and model saving as callbacks, it looks very long and ugly here.

            # User has to choose a few metrics to monitor. Here I chose valid_loss and valid_rmse.
            self.monitored_metric["metric_score"] = torch.clone(valid_rmse)

            if self.early_stopping is not None:
                # TODO: Implement this properly, Add save_model_artifacts here as well.
                best_score, early_stop = self.early_stopping.should_stop(
                    curr_epoch_score=valid_loss
                )
                self.best_valid_loss = best_score

                if early_stop:
                    training_logger.info("Stopping Early!")
                    break
            else:

                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss

                if self.monitored_metric["mode"] == "max":
                    if (
                        self.monitored_metric["metric_score"]
                        > self.best_valid_score
                    ):
                        training_logger.info(
                            f"\nValidation {self.monitored_metric['metric_name']} improved from {self.best_valid_score} to {self.monitored_metric['metric_score']}"
                        )
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]

                else:
                    if (
                        self.monitored_metric["metric_score"]
                        < self.best_valid_score
                    ):
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]
                        training_logger.info(
                            f"\nValidation {self.monitored_metric['metric_name']} improved from {self.best_valid_score} to {self.monitored_metric['metric_score']}"
                        )
                        self.best_valid_score = self.monitored_metric[
                            "metric_score"
                        ]
                        # Reset patience counter as we found a new best score
                        patience_counter_ = self.patience_counter

                        saved_model_path = Path(
                            self.model_path,
                            f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                        )
                        self.save_model_artifacts(
                            saved_model_path,
                            valid_trues,
                            valid_logits,
                            valid_preds,
                            valid_probs,
                        )

                        shutil.copy(
                            saved_model_path.__str__(),
                            os.path.join(
                                self.wandb_run.dir,
                                f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
                            ),
                        )

                        training_logger.info(
                            f"\nSaving model with best valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}"
                        )
                    else:
                        patience_counter_ -= 1
                        training_logger.info(
                            f"Patience Counter {patience_counter_}"
                        )
                        if patience_counter_ == 0:
                            training_logger.info(
                                f"\n\nEarly Stopping, patience reached!\n\nbest valid {self.monitored_metric['metric_name']} score: {self.best_valid_score}"
                            )
                            break
            ########################## End of Early Stopping ############################
            ########################## End of Model Saving ##############################

            ########################## Start of Scheduler ###############################

            if self.scheduler is not None:
                # Special Case for ReduceLROnPlateau
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(self.monitored_metric["metric_score"])
                else:
                    self.scheduler.step()

        ############################## End of Scheduler #################################

        ########################## Load Best Model ######################################
        # Load current checkpoint so we can get model's oof predictions, often in the form of probabilities.
        curr_fold_best_checkpoint = self.load(
            Path(
                self.model_path,
                f"{self.params.model_name}_best_{self.monitored_metric['metric_name']}_fold_{fold}.pt",
            )
        )
        ########################## End of Load Best Model ###############################

        ######################### Delete Optimizer and Scheduler ########################
        utils.free_gpu_memory(
            self.optimizer,
            self.scheduler,
            valid_trues,
            valid_logits,
            valid_preds,
            valid_probs,
        )
        ########################## End of Delete Optimizer and Scheduler ################

        return curr_fold_best_checkpoint

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train one epoch of the model."""

        metric_monitor = metrics.MetricMonitor()

        # set to train mode
        self.model.train()
        average_cumulative_train_loss: float = 0.0
        train_bar = tqdm(train_loader)

        # Iterate over train batches
        # TODO: Consider rename data to batch for names consistency.
        for step, data in enumerate(train_bar, start=1):

            is_mixup = np.random.randint(1, 10) >= 5
            # TODO: Implement MIXUP logic. Refer here: https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix and my https://colab.research.google.com/drive/1sYkKG8O17QFplGMGXTLwIrGKjrgxpRt5#scrollTo=5y4PfmGZubYp
            if self.params.mixup and is_mixup:
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True)
                (
                    inputs,
                    targets_a,
                    targets_b,
                    lam,
                ) = transformation.mixup_data(
                    inputs, targets, params=TRANSFORMS.mixup_params
                )
                inputs, targets_a, targets_b = (
                    inputs.to(self.device, non_blocking=True),
                    targets_a.to(self.device, non_blocking=True),
                    targets_b.to(self.device, non_blocking=True),
                )
            else:
                # unpack and .view(-1, 1) if BCELoss
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True)

            if self.params.use_amp:
                self.optimizer.zero_grad()
                with torch.cuda.amp.autocast(
                    enabled=True, dtype=torch.float16, cache_enabled=True
                ):
                    logits = self.model(inputs)  # Forward pass logits
                    batch_size = inputs.shape[0]
                    if self.params.mixup and is_mixup:
                        curr_batch_train_loss = transformation.mixup_criterion(
                            torch.nn.BCEWithLogitsLoss(),
                            logits,
                            targets_a,
                            targets_b,
                            lam,
                        )
                    else:
                        curr_batch_train_loss = self.train_criterion(
                            targets,
                            logits,
                            batch_size,
                            criterion_params=CRITERION_PARAMS,
                        )
                self.scaler.scale(curr_batch_train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            ## UNCOMMENT FOR GRAD ACCUM
            # if self.params.use_amp:
            #     # self.optimizer.zero_grad()
            #     with torch.cuda.amp.autocast(
            #         enabled=True, dtype=torch.float16, cache_enabled=True
            #     ):
            #         logits = self.model(inputs)  # Forward pass logits
            #         curr_batch_train_loss = self.train_criterion(
            #             targets,
            #             logits,
            #             batch_size,
            #             criterion_params=CRITERION_PARAMS,
            #         )
            #         curr_batch_train_loss /= (
            #             self.params.grad_accumulation_params[
            #                 "iters_to_accumulate"
            #             ]
            #         )
            #     self.scaler.scale(curr_batch_train_loss).backward()
            #     if (step + 1) % self.params.grad_accumulation_params[
            #         "iters_to_accumulate"
            #     ] == 0:
            #         self.scaler.step(self.optimizer)
            #         self.scaler.update()
            #         self.optimizer.zero_grad()
            else:
                logits = self.model(inputs)  # Forward pass logits
                self.optimizer.zero_grad()  # reset gradients
                if self.params.mixup and is_mixup:
                    curr_batch_train_loss = transformation.mixup_criterion(
                        torch.nn.BCEWithLogitsLoss(),
                        logits,
                        targets_a,
                        targets_b,
                        lam,
                    )
                else:
                    curr_batch_train_loss = self.train_criterion(
                        targets,
                        logits,
                        batch_size,
                        criterion_params=CRITERION_PARAMS,
                    )

                curr_batch_train_loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights using the optimizer

            # Update loss metric
            metric_monitor.update("Loss", curr_batch_train_loss.item())
            train_bar.set_description(f"Train. {metric_monitor}")

            average_cumulative_train_loss += (
                curr_batch_train_loss.detach().item()
                - average_cumulative_train_loss
            ) / (step)

            _y_train_prob = self.get_sigmoid_softmax()(logits)
            _y_train_pred = torch.argmax(_y_train_prob, dim=1)

        # TODO: Consider enhancement that returns the same dict as valid_one_epoch.
        return {"train_loss": average_cumulative_train_loss}

    def valid_one_epoch(
        self, valid_loader: torch.utils.data.DataLoader
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Validate the model on the validation set for one epoch.

        Args:
            valid_loader (torch.utils.data.DataLoader): The validation set dataloader.

        Returns:
            Dict[str, np.ndarray]:
                valid_loss (float): The validation loss for each epoch.
                valid_trues (np.ndarray): The ground truth labels for each validation set. shape = (num_samples, 1)
                valid_logits (np.ndarray): The logits for each validation set. shape = (num_samples, num_classes)
                valid_preds (np.ndarray): The predicted labels for each validation set. shape = (num_samples, 1)
                valid_probs (np.ndarray): The predicted probabilities for each validation set. shape = (num_samples, num_classes)
        """

        self.model.eval()  # set to eval mode
        metric_monitor = metrics.MetricMonitor()
        average_cumulative_valid_loss: float = 0.0
        valid_bar = tqdm(valid_loader)

        valid_logits, valid_trues, valid_preds, valid_probs = [], [], [], []

        with torch.no_grad():
            for step, data in enumerate(valid_bar, start=1):
                # unpack
                inputs = data["X"].to(self.device, non_blocking=True)
                targets = data["y"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad()  # reset gradients

                logits = self.model(inputs)  # Forward pass logits

                # get batch size, may not be same as params.batch_size due to whether drop_last in loader is True or False.
                _batch_size = inputs.shape[0]

                # TODO: Refer to my RANZCR notes on difference between Softmax and Sigmoid with examples.
                y_valid_prob = self.get_sigmoid_softmax()(logits)
                y_valid_pred = torch.argmax(y_valid_prob, axis=1)

                curr_batch_val_loss = self.valid_criterion(
                    targets, logits, criterion_params=CRITERION_PARAMS
                )

                metric_monitor.update(
                    "Loss", curr_batch_val_loss.item()
                )  # Update loss metric

                valid_bar.set_description(f"Validation. {metric_monitor}")
                average_cumulative_valid_loss += (
                    curr_batch_val_loss.item() - average_cumulative_valid_loss
                ) / (step)

                valid_trues.extend(targets.cpu())
                valid_logits.extend(logits.cpu())
                valid_preds.extend(y_valid_pred.cpu())
                valid_probs.extend(y_valid_prob.cpu())

        valid_trues, valid_logits, valid_preds, valid_probs = (
            torch.vstack(valid_trues),
            torch.vstack(valid_logits),
            torch.vstack(valid_preds),
            torch.vstack(valid_probs),
        )
        num_valid_samples = len(valid_trues)
        assert valid_trues.shape == valid_preds.shape == (num_valid_samples, 1)
        assert (
            valid_logits.shape
            == valid_probs.shape
            == (num_valid_samples, self.params.num_classes)
        )

        return {
            "valid_loss": average_cumulative_valid_loss,
            "valid_trues": valid_trues,
            "valid_logits": valid_logits,
            "valid_preds": valid_preds,
            "valid_probs": valid_probs,
        }

    def log_metrics(
        self, epoch: int, history: Dict[str, Union[float, np.ndarray]]
    ):
        """Log a scalar value to both MLflow and TensorBoard
        Args:
            history (Dict[str, Union[float, np.ndarray]]): A dictionary of metrics to log.
        """
        for metric_name, metric_values in history.items():
            self.wandb_run.log(
                {metric_name: metric_values[epoch - 1]}, step=epoch
            )

    def log_weights(self, step):
        """Log the weights of the model to both MLflow and TensorBoard.
        # TODO: Check https://github.com/ghnreigns/reighns-mnist/tree/master/reighns_mnist
        Args:
            step ([type]): [description]
        """
        self.writer.add_histogram(
            tag="conv1_weight",
            values=self.model.conv1.weight.data,
            global_step=step,
        )

    def save_model_artifacts(
        self,
        path: str,
        valid_trues: torch.Tensor,
        valid_logits: torch.Tensor,
        valid_preds: torch.Tensor,
        valid_probs: torch.Tensor,
    ) -> None:
        """Save the weight for the best evaluation metric and also the OOF scores.

        Caution: I removed model.eval() here as this is not standard practice.

        valid_trues -> oof_trues: np.array of shape [num_samples, 1] and represent the true labels for each sample in current fold.
                                i.e. oof_trues.flattened()[i] = true label of sample i in current fold.
        valid_logits -> oof_logits: np.array of shape [num_samples, num_classes] and represent the logits for each sample in current fold.
                                i.e. oof_logits[i] = [logit_of_sample_i_in_current_fold_for_class_0, logit_of_sample_i_in_current_fold_for_class_1, ...]
        valid_preds -> oof_preds: np.array of shape [num_samples, 1] and represent the predicted labels for each sample in current fold.
                                i.e. oof_preds.flattened()[i] = predicted label of sample i in current fold.
        valid_probs -> oof_probs: np.array of shape [num_samples, num_classes] and represent the probabilities for each sample in current fold. i.e. first row is the probabilities of the first class.
                                i.e. oof_probs[i] = [probability_of_sample_i_in_current_fold_for_class_0, probability_of_sample_i_in_current_fold_for_class_1, ...]
        """

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "oof_trues": valid_trues,
                "oof_logits": valid_logits,
                "oof_preds": valid_preds,
                "oof_probs": valid_probs,
            },
            path,
        )

    @staticmethod
    def load(path: str):
        """Load a model checkpoint from the given path.
        Reason for using a static method: https://stackoverflow.com/questions/70052073/am-i-using-static-method-correctly/70052107#70052107
        """
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        return checkpoint
