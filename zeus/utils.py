import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from zeus.configs import GMMConfig, InferenceMethodType, MetricType
from zeus.wandb_logging import accumulate_batch_logs, log_epoch
from inference_methods.simple_gmm import SimplifiedGMM


def hungarian_algorithm(A):
    # A is a torch.float32 tensor of shape (n, m)
    n, m = A.shape
    INF = 1e9  # a large number used as infinity

    # Create arrays with an extra dummy index at 0.
    # u and v are the dual variables.
    u = torch.zeros(n + 1, dtype=torch.float32)
    v = torch.zeros(m + 1, dtype=torch.float32)
    # p and way store matching information.
    p = torch.zeros(m + 1, dtype=torch.long)  # p[j] will eventually hold the row (1-indexed) matched with column j
    way = torch.zeros(m + 1, dtype=torch.long)

    # Loop over rows, using 1-indexing for the algorithm
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        # Initialize minv and used arrays (dummy index at 0)
        minv = torch.full((m + 1,), INF, dtype=torch.float32)
        used = torch.zeros(m + 1, dtype=torch.bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])  # current row (1-indexed)
            delta = INF
            j1 = 0
            # Iterate over columns 1...m (1-indexed)
            for j in range(1, m + 1):
                if not used[j]:
                    # Adjust indices for A: row i0 corresponds to A[i0-1], and column j corresponds to A[:, j-1]
                    cur = A[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j].item()
                        j1 = j
            # Update dual variables and minv values
            for j in range(m + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Reconstruct the matching
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    # Build the final assignment:
    # For each column j (1-indexed), p[j] is the row (1-indexed) matched to j.
    # We convert this into a 0-indexed assignment array where:
    # assignment[i] = j means row i is assigned to column j.
    assignment = torch.full((n,), -1, dtype=torch.long)
    for j in range(1, m + 1):
        if p[j] != 0:
            # Convert both row and column to 0-index
            assignment[int(p[j]) - 1] = j - 1
    return assignment


def gmm_loss_with_regularizes(output, y_batch, *, probs):
    output = output.squeeze(1)
    num_classes = len(torch.unique(y_batch))
    one_hot = torch.eye(num_classes, device=y_batch.device)[y_batch]

    cluster_sums = one_hot.T @ output
    cluster_counts = one_hot.sum(dim=0)
    cluster_means = cluster_sums / cluster_counts.unsqueeze(-1)

    all_distances = -torch.linalg.norm(output.unsqueeze(1) - cluster_means.unsqueeze(0), dim=-1) ** 2
    probs = probs.to(output.device)

    # exp_dist = torch.exp(all_distances - torch.max(all_distances, dim=-1, keepdim=True).values)
    # prob_soft = exp_dist*probs / torch.sum(exp_dist*probs, dim=-1, keepdim=True)
    # total_loss = -torch.log(prob_soft+1e-10)[torch.arange(len(output)), y_batch].mean()
    total_loss = -torch.log_softmax(probs*all_distances, dim=-1)[torch.arange(len(output)), y_batch].mean()
    # total_loss = -torch.log_softmax(all_distances, dim=-1)[torch.arange(len(output)), y_batch].mean()

    dist_lambda = 1.0
    # if cur_epoch > 20:
    #     dist_lambda = min(1, (cur_epoch - 20) / 20)

    if dist_lambda:
        cluster_diffs = cluster_means.unsqueeze(1) - cluster_means.unsqueeze(0)
        cluster_distances = torch.minimum(torch.linalg.norm(cluster_diffs, dim=-1) ** 2, torch.tensor(0.5))

        total_loss -= dist_lambda * cluster_distances.mean()

        means = cluster_means[y_batch]
        distances = torch.linalg.norm(output - means, dim=-1) ** 2
        total_loss += dist_lambda * distances.mean()

    return total_loss


def evaluate_model(model, test_datasets, config: GMMConfig, dataset_type, batch_log=True, save_results=False):
    print("\n\n Dataset type: ", dataset_type)
    log_prefix = f"{dataset_type}/"

    run_results = {}
    model.eval()

    metrics = defaultdict(lambda: defaultdict(int))
    for batch_idx, dataset_data in enumerate(test_datasets):
        if "real" not in dataset_type:
            X_batch, y_batch, mode, X_true, _ = dataset_data
            log_prefix_batch = log_prefix
        else:
            X_batch, y_batch, *_ = dataset_data
            mode = ""
            log_prefix_batch = f"{dataset_type}_{batch_idx}/"
        n_clusters = len(np.unique(y_batch))

        x_dim = X_batch.shape[1]

        if x_dim > config.dim:
            pca = PCA(n_components=config.pca_dim)
            X_batch = pca.fit_transform(X_batch)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X_batch = scaler.fit_transform(X_batch)
            X_batch = torch.tensor(X_batch, dtype=torch.float32)
        elif x_dim < config.dim:
            zeros = torch.zeros(X_batch.shape[0], config.dim-X_batch.shape[1])
            X_batch = torch.cat((X_batch, zeros), dim=1)

        X_batch, y_batch = X_batch.unsqueeze(1).to(config.device), y_batch.to(config.device)
        with torch.no_grad():
            output = model(X_batch, k=n_clusters)
            output, centers = output[:-config.num_gaussians], output[-config.num_gaussians:]

        if config.metric_type == MetricType.BRIER:
            output = output.squeeze(1).cpu()
            scaler = MinMaxScaler((-1, 1))
            output = scaler.fit_transform(output)
            gmm = SimplifiedGMM(n_components=n_clusters, n_init=100)
            gmm.fit(output)

            probs = gmm.predict_proba(output)
            probs = torch.tensor(probs, dtype=torch.float32, device=config.device)
            assignments = true_class_assignments(probs, y_batch)

            probs_true = probs[:, assignments]
            y_ohe = torch.nn.functional.one_hot(y_batch, num_classes=n_clusters)
            metric = torch.mean((probs_true - y_ohe)**2, dim=0).sum()
        else:
            y_pred = predict_clusters(output, n_clusters, config.device, inf_method=config.inf_method, n_init=100)

            metric, _ = accumulate_batch_logs(y_pred, y_batch, metrics, mode, cur_dim=x_dim,
                                              batch_logging=batch_log, log_prefix=log_prefix_batch, model_name="test")

        run_results[f"dataset-{batch_idx}"] = [metric.item()]

    if save_results:
        run_results["checkpoint"] = [config.model_path]
        run_results = pd.DataFrame(run_results)
        print(run_results)
        df_saved = (
            pd.read_csv(config.results_file)
            if os.path.exists(config.results_file) and os.path.getsize(config.results_file) > 0
            else pd.DataFrame()
        )
        df_combined = pd.concat([df_saved, run_results], ignore_index=True)
        df_combined.to_csv(
            config.results_file, index=False, float_format="%.4f"
        )

    log_dict = log_epoch(metrics, "", "", log_prefix, True)
    wandb.log(log_dict)


def predict_for_centers(output, centers):
    all_distances = -torch.linalg.norm(output.unsqueeze(1) - centers.unsqueeze(0), dim=-1) ** 2
    y_pred = torch.argmax(all_distances, dim=-1)

    return y_pred


def predict_clusters(output, n_clusters, device, n_init=1, inf_method=InferenceMethodType.KMEANS):
    output = output.squeeze(1).detach()
    output = output.cpu()
    scaler = MinMaxScaler((-1, 1))
    output = scaler.fit_transform(output)

    if inf_method == InferenceMethodType.GMM:
        gmm = GaussianMixture(n_components=n_clusters, n_init=int(n_init/10))
        y_pred = torch.from_numpy(gmm.fit_predict(output))
    elif inf_method == InferenceMethodType.KMEANS:
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit_transform(output)

        y_pred = kmeans.labels_
        y_pred = torch.tensor(y_pred)
    else:  # out_method == InferenceMethodType.SIMPLE_GMM
        gmm = SimplifiedGMM(n_components=n_clusters, n_init=10)
        gmm.fit(output)

        y_pred = gmm.predict(output)
        y_pred = torch.tensor(y_pred)

    return y_pred.to(device)


def true_class_assignments(probs: torch.Tensor, y: torch.Tensor):
    n_classes = len(torch.unique(y))
    y_ohe = torch.nn.functional.one_hot(y, num_classes=n_classes)

    cost_matrix = -torch.sum(probs.unsqueeze(1)*y_ohe.unsqueeze(-1), dim=0)
    true_assignments = hungarian_algorithm(cost_matrix)
    return true_assignments


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


openml_ids = [14, 15, 16, 18, 22, 458, 1510, 35, 53, 56, 61, 187, 377, 481, 694, 721, 733, 745, 756, 796,
              820, 840, 854, 1495, 1499, 1523, 4153, 40496, 40682, 40705, 42261, 42585, 1462, 51]
