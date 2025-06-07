import torch
import wandb
from torchmetrics.functional.clustering import normalized_mutual_info_score, adjusted_rand_score


def accumulate_batch_logs(preds, y_batch, metrics, mode, batch_logging=False, log_prefix="", model_name="", cur_dim=0):
    preds = torch.tensor(preds) if type(preds) is not torch.Tensor else preds
    nmi = normalized_mutual_info_score(preds, y_batch)
    ari = adjusted_rand_score(preds, y_batch)

    metrics[mode]["nmi"] += nmi
    metrics[mode]["ari"] += ari
    metrics[mode]["batches"] += 1

    if batch_logging:
        print('\n', model_name)
        print("NMI: ", nmi)
        print("ARI: ", ari)
        wandb.log({
            f"{log_prefix}{model_name}_nmi": nmi,
            f"{log_prefix}{model_name}_ari": ari,
            f"{log_prefix}{model_name}_dim": cur_dim,
        })

    return ari, nmi


def log_epoch(metrics, model_name, dataset_type, log_prefix, log_nmi=False):
    for mode_name in metrics.keys():
        metrics[mode_name]["nmi"] /= metrics[mode_name]["batches"]
        metrics[mode_name]["ari"] /= metrics[mode_name]["batches"]

    log_dict = {}
    for mode_name, metric_values in metrics.items():
        for metric_name, metric_value in metric_values.items():
            if metric_name == "batches" or (metric_name == "nmi" and not log_nmi):
                print("Batches", metric_value)
                continue
            print(f"{dataset_type} {model_name} {metric_name} baseline: {mode_name} {metric_value:.4f}")
            log_dict[f"{log_prefix}{dataset_type}_{model_name}_{mode_name}_{metric_name}_epoch"] = metric_value

    return log_dict
