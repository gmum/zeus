from typing import cast, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from zeus.configs import GMMConfig
from zeus.model import encoders
from zeus.model.zeus import ZeusTransformerModel
from zeus.utils import setup_seed


def config_initialization() -> GMMConfig:
    base_config = OmegaConf.structured(GMMConfig)
    arg_config = OmegaConf.from_cli()

    config = cast(GMMConfig, OmegaConf.merge(base_config, arg_config))
    config.pca_dim = config.dim if config.pca_dim == 0 else config.pca_dim
    print(OmegaConf.to_yaml(config))

    return config


def model_initialization(config: GMMConfig) -> ZeusTransformerModel:
    encoder = encoders.Linear(config.dim, config.embed_dim, replace_nan_by_zero=True)
    model = ZeusTransformerModel(
        encoder, config.num_gaussians, config.embed_dim,
        config.n_head, config.hid_dim, config.n_layers, config.dropout, efficient_eval_masking=True,
        n_clusters=config.num_gaussians, dist_based_logit=config.dist_based_logit, loss_type=config.loss_type,
    )
    model = model.to(config.device)
    return model


def initialize() -> Tuple[GMMConfig, ZeusTransformerModel]:
    setup_seed(42)

    config = config_initialization()

    model = model_initialization(config)
    if config.model_path != "":
        checkpoint = torch.load(config.model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model"], strict=False)
    print(f"Params ", np.sum([p.numel() for p in model.parameters()]) / 1e6, "M")

    return config, model
