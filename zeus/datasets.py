from itertools import islice

import numpy as np
import openml
import torch
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from scipy.linalg import sqrtm
from scipy.sparse import issparse
from torch import nn


# Define random neural network
class SpectralNormBlock(nn.Module):
    def __init__(self, dim, h=0.5):
        super().__init__()
        self.h = h
        self.dim = dim

        # Matrix B: input -> hidden
        hid_dim = 4 * dim
        B = torch.randn((dim, hid_dim))
        U, S, U_T = torch.linalg.svd(B, full_matrices=False)
        S_new = torch.rand(*S.shape) * 0.5 + 0.5
        S_new = torch.sort(S_new)[0].flip(dims=[0])
        self.B = U @ torch.diag(S_new) @ U_T

        # Matrix A: hidden -> output
        A = torch.randn((hid_dim, dim))
        U, S, U_T = torch.linalg.svd(A, full_matrices=False)
        S_new = torch.rand(*S.shape) * 0.25 + 0.75
        S_new = torch.sort(S_new)[0].flip(dims=[0])
        self.A = U @ torch.diag(S_new) @ U_T

    def forward(self, x, num_blocks):
        # f(x) = x + h * A * ReLU(B * x)
        out = x @ self.B
        out = torch.nn.functional.relu(out)
        out = out @ self.A
        return x + out / (torch.max(torch.std(out, dim=0, keepdim=True)))


class RandomNetwork(nn.Module):
    def __init__(self, dim, n_components, num_blocks=3, h=0.5):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpectralNormBlock(dim + n_components, h=h)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x, len(self.blocks))
            x = (x - torch.mean(x, dim=0, keepdim=True)) / (torch.std(x, dim=0, keepdim=True) + 1e-8)
        return x


def create_gaussian_mixture(
        num_gaussians, min_points, max_points, dim, min_distance,
        p1, p2, *, num_categorical=5, max_categories=5, start_distance,
        categorical_chance, only_categorical
):
    data = []
    labels = []
    means, covs = [], []
    points = []

    # Generate between 2 and num_gaussians clusters
    #if np.random.rand() < 0.0:
    #    dataset_gaussians = 2
    #else:
    dataset_gaussians = np.random.randint(2, num_gaussians + 1)

    # Ensure reasonable number of points per cluster
    cluster_max_points = min(max_points, 2500 // dataset_gaussians + 1)
    # if np.random.rand() < 0.25:
    #    cluster_max_points = 200

    categorical_dims = []
    total_cat_dims = 0

    if np.random.rand() < categorical_chance or only_categorical:
        cur_categorical = np.random.randint(num_categorical) + 1
        cur_categorical = cur_categorical if not only_categorical else cur_categorical + 5

        for _ in range(cur_categorical):
            n_categories = np.random.randint(2, max_categories + 1)
            categorical_dims.append(n_categories)
            total_cat_dims += n_categories

    # Ensure continuous dimensions + categorical dimensions <= dim
    if not only_categorical:
        max_continuous_dim = dim - total_cat_dims
        if max_continuous_dim < 2:
            cur_dim = 2  # Minimum 2D for continuous features
            categorical_dims = []  # No categorical features
            total_cat_dims = 0
        else:
            cur_dim = np.random.randint(2, min(max_continuous_dim + 1, dim + 1))
            # cur_dim = dim
    else:
        cur_dim = 0

    for gaussian_index in range(dataset_gaussians):
        cur_min_distance = min_distance if start_distance is None \
            else np.random.uniform(min_distance, start_distance)
        points_per_gaussian = np.random.randint(min_points, cluster_max_points + 1)

        if not only_categorical:
            random_matrix = np.random.randn(cur_dim, cur_dim)
            covariance_matrix = np.dot(random_matrix, random_matrix.T)

            cur_p2 = p2
            cur_p1 = p1 if np.random.rand() > 0.75 else p2 - p1

            U, S, U_T = np.linalg.svd(covariance_matrix)
            S_new = np.random.rand(*S.shape) * (cur_p2 - cur_p1) + cur_p1

            if np.random.rand() < 0.2:
                zero_out_dims = max(1, S_new.shape[0] // 3)
                chosen_idx = np.random.randint(zero_out_dims) + 1
                S_new[:chosen_idx] = np.random.rand(chosen_idx) * p1
            S_new = np.sort(S_new)[::-1]

            cov = U @ np.diag(S_new) @ U_T

            sqrt_cov = sqrtm(cov)

            traces = [
                np.trace(cov + cov_j[0] - 2 * sqrtm(sqrt_cov @ cov_j[0] @ sqrt_cov))
                for cov_j in covs
            ]

            # Generate means with minimum Wasserstein-2 distance
            # i = 100
            found_mean = False
            retry_steps = 1
            while True:
                mean = torch.zeros(size=(cur_dim,))
                mean_dir = np.random.uniform(-1, 1, size=cur_dim)
                mean_dir = mean_dir / np.linalg.norm(mean_dir)
                eps = 0.003 * max(1, (50 - cur_dim + 1)) * retry_steps * cur_min_distance
                i = 500
                while i > 0:
                    mean += eps * mean_dir
                    i -= 1
                    if all(
                            trace + np.linalg.norm(mean - m) ** 2 > cur_min_distance
                            for m, trace in zip(means, traces)
                    ):
                        found_mean = True
                        break

                retry_steps *= 2
                if found_mean:
                    break

            points.append(points_per_gaussian)
            gaussian_data = np.random.multivariate_normal(mean, cov, points_per_gaussian)

            means.append(mean)
            covs.append(cov[np.newaxis, :])
        else:
            gaussian_data = np.empty((points_per_gaussian, 0))

        categorical_data = []
        for n_categories in categorical_dims:
            # Bias towards certain categories for each Gaussian
            probs = np.random.dirichlet(alpha=[0.5] * n_categories)
            cat_values = np.random.choice(n_categories, size=points_per_gaussian, p=probs)
            categorical_data.append(cat_values)

        # Convert to one-hot encoding
        categorical_one_hot = []
        for cat_values, n_categories in zip(categorical_data, categorical_dims):
            one_hot = np.eye(n_categories)[cat_values]
            categorical_one_hot.append(one_hot)

        # Combine continuous and categorical features
        combined_data = np.hstack([gaussian_data] + [cat_data for cat_data in categorical_one_hot])
        data.append(combined_data)

        labels.extend([gaussian_index] * points_per_gaussian)

    X = np.vstack(data)

    if not only_categorical:
        std = StandardScaler()
        min_scaler = MinMaxScaler(feature_range=(-1, 1))
        X[:, :cur_dim] = min_scaler.fit_transform(std.fit_transform(X[:, :cur_dim]))
        # X[:, :cur_dim] = std.fit_transform(X[:, :cur_dim])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    points = torch.tensor(points, dtype=torch.float32)
    # More stable probability calculation
    probs = points / torch.sum(points)
    total_dim = cur_dim + total_cat_dims

    return X, y, total_dim, cur_dim, probs


def dataset_generator(num_gaussians, min_points=200, max_points=400, dim=2, min_distance=1.5,
                      p1=0.25, p2=1.0, mode="random", max_blocks=6, *,
                      num_categorical=5, max_categories=5, start_distance=None,
                      categorical_chance=0.3, only_categorical=False):

    while True:
        X, y, total_dim, cur_dim, probs = create_gaussian_mixture(
            num_gaussians, min_points, max_points, dim, min_distance, p1, p2, num_categorical=num_categorical,
            max_categories=max_categories, start_distance=start_distance, categorical_chance=categorical_chance,
            only_categorical=only_categorical
        )
        gen_mode = "gaussian"

        if mode == "random":
            gen_mode = f"{gen_mode}_transformed" if np.random.rand() < 0.5 else gen_mode
        else:
            gen_mode = f"{gen_mode}_{mode}"

        if "transformed" in gen_mode:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X[:, :cur_dim] = torch.tensor(scaler.fit_transform(X[:, :cur_dim]), dtype=torch.float32)

            h = np.random.uniform(0.1, 0.9)  # More stable h range
            min_blocks = 3

            cur_gaussians = len(np.unique(y))
            blocks = np.random.randint(min_blocks, min_blocks+max_blocks)
            net = RandomNetwork(cur_dim, cur_gaussians, num_blocks=blocks, h=h)

            y_one_hot = torch.zeros(y.shape[0], cur_gaussians)
            y_one_hot[torch.arange(y.shape[0]), y] = 1
#
            # Split continuous and categorical features
            X_continuous = X[:, :cur_dim]
            X_categorical = X[:, cur_dim:]
#
            X_with_classes = torch.cat((X_continuous, y_one_hot), dim=1)
            X_transformed = net(X_with_classes)
            # cur_pca_dim = np.random.randint(cur_dim, cur_dim+min(cur_gaussians, dim-total_dim+1))
            # total_dim += cur_pca_dim - cur_dim
            cur_pca_dim = cur_dim
            pca = PCA(n_components=cur_pca_dim)
            X_transformed = pca.fit_transform(X_transformed)

            if cur_dim > 0:
                std = StandardScaler()
                min_scaler = MinMaxScaler(feature_range=(-1, 1))
                X_transformed = min_scaler.fit_transform(std.fit_transform(X_transformed))
                #X_transformed = std.fit_transform(X_transformed)

            # pad X_transformed with zeros
            X_transformed = torch.cat((torch.tensor(X_transformed, dtype=torch.float32), X_categorical,
                                       torch.zeros(X_transformed.shape[0], dim - total_dim)), dim=1)

            yield X_transformed, y, gen_mode, X, probs
        else:
            # pad X with zeros
            X = torch.cat((X, torch.zeros(X.shape[0], dim - total_dim)), dim=1)

            yield X, y, gen_mode, X, probs


class GMMDataset(torch.utils.data.Dataset):
    def __init__(self, num_datasets, num_gaussians, min_points=200, max_points=400, dim=2, min_distance=1.5,
                 p1=0.25, p2=1.0, mode="random", max_blocks=6, *,
                 num_categorical=5, max_categories=5, start_distance=None,
                 categorical_chance=0.3, only_categorical=False):
        self.num_datasets = num_datasets
        self.num_gaussians = num_gaussians
        self.min_points = min_points
        self.max_points = max_points
        self.dim = dim
        self.min_distance = min_distance
        self.p1 = p1
        self.p2 = p2
        self.mode = mode
        self.max_blocks = max_blocks
        self.num_categorical = num_categorical
        self.max_categories = max_categories
        self.start_distance = start_distance
        self.categorical_chance = categorical_chance
        self.only_categorical = only_categorical

        # Set random seed based on dataset index to ensure reproducibility
        self.base_seed = np.random.randint(0, 10000)

    def __len__(self):
        return self.num_datasets

    def __getitem__(self, idx):
        # Set seed for this specific dataset

        return list(islice(
            dataset_generator(
                self.num_gaussians, self.min_points, self.max_points, self.dim, self.min_distance, self.p1, self.p2,
                self.mode, self.max_blocks, num_categorical=self.num_categorical,
                start_distance=self.start_distance, categorical_chance=self.categorical_chance,
                only_categorical=self.only_categorical), 1)
        )[0]


def generate_gmm_datasets_with_projected_points(num_datasets, num_gaussians, min_points=200, max_points=400, dim=2,
                                                min_distance=1.5, p1=0.25, p2=1.0, *,
                                                num_categorical, max_categories, categorical_chance=0.3,
                                                max_blocks=6, gen_mode="random",
                                                start_distance=None):
    datasets = list(islice(dataset_generator(num_gaussians, min_points, max_points, dim,
                                             min_distance, p1, p2, mode=gen_mode,
                                             num_categorical=num_categorical, max_categories=max_categories,
                                             categorical_chance=categorical_chance, max_blocks=max_blocks,
                                             start_distance=start_distance),
                           num_datasets))
    return datasets


def load_real_datasets(input_size, openml_idx=None, use_pca=False, pca_dim=2, *, return_whole_dataset):

    datasets = []
    statistics = defaultdict(list)
    for id in openml_idx:
        id = int(id)
        dataset = openml.datasets.get_dataset(id, download_data=False)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute)

        categorical_cols = [name for name, is_cat in zip(attribute_names, categorical_indicator) if is_cat]
        numerical_cols = [name for name, is_cat in zip(attribute_names, categorical_indicator) if not is_cat]

        num_imputer = SimpleImputer(strategy="mean")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', num_imputer),
                    ('scaler', StandardScaler()),
                    ('min_max', MinMaxScaler(feature_range=(-1, 1)))
                ]), numerical_cols),
                ('cat', Pipeline(steps=[
                    ('imputer', cat_imputer),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_cols)
            ]
        )

        # Apply preprocessing
        X = preprocessor.fit_transform(X)
        if issparse(X):
            X = X.toarray()

        le = LabelEncoder()
        y = le.fit_transform(y)

        #print("\n\nDataset id: ", id)
        #print("Dataset size: ", len(X))
        #print("Len num: ", len(numerical_cols))
        #print("Len cat: ", len(categorical_cols))
        #print("Dim: ", X.shape[1])
        #print("Cat one-hot: ", X.shape[1]-len(numerical_cols))
        #print("Classes: ", len(np.unique(y)))

        statistics["id"].append(id)
        statistics["instances"].append(len(X))
        statistics["Num features"].append(len(numerical_cols))
        statistics["Cat features"].append(len(categorical_cols))
        statistics["Dim"].append(X.shape[1])
        statistics["Cat one-hot"].append(X.shape[1]-len(numerical_cols))
        statistics["Classes"].append(len(np.unique(y)))

        # if X.shape[1] > pca_dim + 5:
        #     continue
        if return_whole_dataset:
            datasets.append(
                (
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(np.array(y), dtype=torch.int64),
                    input_size,
                    len(numerical_cols)
                )
            )
            # print("Returned whole, dataset dim: ", X.shape[1])
            continue

        # X = torch.tensor(X, dtype=torch.float32)
        if use_pca and X.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim)
            X = pca.fit_transform(X)
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X = scaler.fit_transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.int64)

        cur_dim = X.shape[1]
        #print("Dataset dim: ", cur_dim)
        X = torch.concatenate((X, torch.zeros(X.shape[0], input_size - cur_dim)), dim=1)

        datasets.append((X, y, input_size, len(numerical_cols)))
    #statistics = pd.DataFrame(statistics)
    #statistics.to_csv('statistics.csv', index=False)
    #print(statistics.to_latex(index=False))
    return datasets
