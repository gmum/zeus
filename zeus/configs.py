from dataclasses import dataclass
from enum import Enum


class LossType(Enum):
    PAIR = 0,
    PROBA = 1,
    CENTER = 2


class InferenceMethodType(Enum):
    KMEANS = 0,
    GMM = 1,
    SIMPLE_GMM = 2


class MetricType(Enum):
    ARI = 0,
    BRIER = 1


class EvalDatasetType(Enum):
    OPENML = 0,
    SYN_GAUSSIAN = 1,
    SYN_TRANSFORMED = 2


@dataclass
class GMMConfig:
    output_dir: str = 'results'
    results_file: str = 'eval.csv'
    gen_mode: str = 'random'
    model_path: str = ""
    inf_method: InferenceMethodType = InferenceMethodType.KMEANS
    metric_type: MetricType = MetricType.ARI
    eval_dataset: EvalDatasetType = EvalDatasetType.OPENML

    nr_epochs: int = 300
    data_scaling: bool = True
    learning_rate: float = 3e-5
    device: str = 'cuda:0'

    use_pca: bool = True
    return_whole_dataset: bool = True

    num_gaussians: int = 10
    min_points: int = 50
    max_points: int = 500
    dim: int = 30
    pca_dim: int = 30
    max_blocks: int = 3
    num_categorical: int = 5
    max_categories: int = 5
    categorical_chance: float = 0.3
    only_categorical: bool = False

    loss_type: LossType = LossType.CENTER

    start_distance: float = 1.0
    end_distance: float = 0.5
    eigenvalue_p1: float = 0.005
    eigenvalue_p2: float = 0.05

    num_train_datasets: int = 1000
    num_test_datasets: int = 200

    embed_dim: int = 512
    n_head: int = 4
    hid_dim: int = 1024
    n_layers: int = 12
    use_cluster_info: bool = False
    dist_based_logit: bool = False
    dropout: float = 0.0
    use_centers: bool = False
