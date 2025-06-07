import torch
import wandb

from zeus.configs import EvalDatasetType
from zeus.datasets import load_real_datasets
from zeus.initialziation import initialize
from zeus.utils import openml_ids, evaluate_model

config, model = initialize()
wandb.init(project='zeus_eval', tags=["ZEUS"], dir=config.output_dir)

dataset_type = "synthetic"
if config.eval_dataset == EvalDatasetType.OPENML:
    datasets = load_real_datasets(config.dim, openml_ids, use_pca=True,
                                 pca_dim=config.pca_dim, return_whole_dataset=config.return_whole_dataset)
    dataset_type = "real_openml"
elif config.eval_dataset == EvalDatasetType.SYN_GAUSSIAN:
    datasets = torch.load("synthetic_datasets/val_gaussian.pt")
else:  # config.eval_dataset == EvalDatasetType.SYN_TRANSFORMED
    datasets = torch.load("synthetic_datasets/val_transformed.pt")

evaluate_model(model, datasets, config, dataset_type, batch_log=True, save_results=True)

wandb.finish()

