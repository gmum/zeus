import os
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans

import wandb
from omegaconf import OmegaConf

from zeus.datasets import generate_gmm_datasets_with_projected_points, load_real_datasets, GMMDataset
from zeus.initialziation import initialize
from zeus.utils import evaluate_model, predict_clusters, openml_ids, gmm_loss_with_regularizes
from zeus.wandb_logging import accumulate_batch_logs, log_epoch
from zeus.model.model_utils import get_cosine_schedule_with_warmup
from collections import defaultdict

if __name__ == '__main__':
    config, model = initialize()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(config.output_dir, "models", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    val_datasets = generate_gmm_datasets_with_projected_points(
        config.num_test_datasets, config.num_gaussians, config.min_points, config.max_points, config.dim,
        config.end_distance, config.eigenvalue_p1, config.eigenvalue_p2,
        num_categorical=config.num_categorical, max_categories=config.max_categories,
        categorical_chance=config.categorical_chance, max_blocks=config.max_blocks, gen_mode=config.gen_mode,
        start_distance=config.start_distance
    )

    test_datasets = generate_gmm_datasets_with_projected_points(
        config.num_test_datasets, config.num_gaussians, config.min_points, config.max_points, config.dim,
        config.end_distance, config.eigenvalue_p1, config.eigenvalue_p2,
        num_categorical=config.num_categorical, max_categories=config.max_categories,
        categorical_chance=config.categorical_chance, max_blocks=config.max_blocks,
        gen_mode=config.gen_mode, start_distance=config.start_distance
    )

    openml_datasets = load_real_datasets(config.dim, openml_ids, use_pca=True,
                                         pca_dim=config.pca_dim, return_whole_dataset=config.return_whole_dataset)

    wandb.init(project='zeus', tags=["ZEUS"], dir=config.output_dir)
    wandb.config.update(OmegaConf.to_container(config))

    # Calculate KMeans performance on validation datasets
    metrics = defaultdict(lambda: defaultdict(int))
    kmeans_total_time = 0
    for X_batch, y_batch, mode, _, _ in val_datasets:
        n_clusters = len(torch.unique(y_batch))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        X_numpy = X_batch.numpy()

        start_time = time.time()
        kmeans_pred = kmeans.fit_predict(X_numpy)
        kmeans_total_time += time.time() - start_time

        accumulate_batch_logs(kmeans_pred, y_batch, metrics, mode)

    avg_kmeans_time = kmeans_total_time / len(val_datasets)
    print(f"Average KMeans inference time: {avg_kmeans_time:.4f} seconds")

    log_dict = log_epoch(metrics, "kmeans", "val", "")
    wandb.log(log_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, config.nr_epochs*config.num_train_datasets)
    if config.model_path != "":
        checkpoint = torch.load(config.model_path, map_location=torch.device('cpu'))
        optimizer.load_state_dict(checkpoint["optimizer"])

    train_dataset = GMMDataset(config.num_train_datasets, config.num_gaussians, config.min_points, config.max_points,
                               config.dim, config.end_distance, config.eigenvalue_p1, config.eigenvalue_p2,
                               config.gen_mode, config.max_blocks,
                               num_categorical=config.num_categorical, max_categories=config.max_categories,
                               categorical_chance=config.categorical_chance,
                               start_distance=config.start_distance
                               )
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, pin_memory=True,
                              num_workers=8, persistent_workers=True)

    for i in tqdm(range(config.nr_epochs)):
        model.train()

        epoch_loss = 0
        train_metrics = defaultdict(lambda: defaultdict(int))

        for batch_idx, (X_batch, y_batch, mode, _, probs) in enumerate(train_loader):
            # batch_size equals 1
            n_clusters = len(torch.unique(y_batch))
            X_batch, y_batch, mode = X_batch[0], y_batch[0], mode[0]
            X_batch, y_batch = X_batch.unsqueeze(1).to(config.device), y_batch.to(config.device)

            optimizer.zero_grad()

            output = model(X_batch)
            output = output[:-config.num_gaussians]

            loss = gmm_loss_with_regularizes(output, y_batch, probs=probs)

            loss.backward()
            optimizer.step()

            scheduler.step()
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

            y_pred = predict_clusters(output, n_clusters, config.device,
                                      n_init=1, inf_method=config.inf_method).squeeze(-1).detach()
            epoch_loss += loss.item()

            accumulate_batch_logs(y_pred, y_batch, train_metrics, mode)

        epoch_loss /= config.num_train_datasets
        train_log_dict = log_epoch(train_metrics, "", "train", "")

        wandb.log({"train_loss": epoch_loss})
        wandb.log(train_log_dict)

        if i % 25 == 0:
            model.eval()
            val_loss = 0
            model_total_time = 0

            val_metrics = defaultdict(lambda: defaultdict(int))
            for batch_idx, (X_batch, y_batch, mode, _, probs) in enumerate(val_datasets):
                n_clusters = len(torch.unique(y_batch))
                X_batch, y_batch = X_batch.unsqueeze(1).to(config.device), y_batch.to(config.device)

                start_time = time.time()
                with torch.no_grad():
                    output = model(X_batch)
                model_total_time += time.time() - start_time

                output, _ = output[:-config.num_gaussians], output[-config.num_gaussians:]

                val_loss += gmm_loss_with_regularizes(output, y_batch, probs=probs)
                y_pred_ = predict_clusters(output, n_clusters, config.device,
                                           inf_method=config.inf_method, n_init=10).squeeze(-1).detach()
                accumulate_batch_logs(y_pred_, y_batch, val_metrics, f"{mode}")

            val_loss /= len(val_datasets)
            avg_model_time = model_total_time / len(val_datasets)
            val_log_dict = log_epoch(val_metrics, "", "val", "")

            wandb.log({
                "val_loss": val_loss, "val_model_avg_time": avg_model_time
            })
            wandb.log(val_log_dict)

            evaluate_model(model, openml_datasets, config, f"real_openml",
                           batch_log=True, save_results=False)

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}_epoch.pt")
            checkpoint = {
                "model": model.state_dict(),
                "epoch": i,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)

    evaluate_model(model, test_datasets, config, "val")
    evaluate_model(model, openml_datasets, config, f"real_openml",
                   batch_log=True, save_results=False)

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)

    wandb.finish()
