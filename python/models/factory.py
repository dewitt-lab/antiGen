"""Model factory functions for creating and loading fitness predictors."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch

from config import ExperimentConfig, ModelConfig, ModelType, AlphabetConfig, TransformerConfig
from models.base import BaseFitnessPredictor


def create_model(
    config: ModelConfig,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseFitnessPredictor:
    """Factory function to create fitness prediction models.

    Args:
        config: Model configuration specifying architecture and hyperparameters.
        device: Target device for the model. If None, uses CUDA if available.

    Returns:
        Initialized model instance moved to the specified device.

    Raises:
        ValueError: If model_type is not recognized.

    Example:
        >>> from config import ModelConfig, ModelType
        >>> config = ModelConfig(model_type=ModelType.ESM2_FITNESS_PREDICTOR)
        >>> model = create_model(config, device="cuda")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if config.model_type == ModelType.FITNESS_PREDICTOR:
        from models.fitness_predictor import FitnessPredictor
        model = FitnessPredictor(config)
    elif config.model_type == ModelType.ESM2_FITNESS_PREDICTOR:
        from models.esm2_fitness_predictor import ESM2FitnessPredictor
        model = ESM2FitnessPredictor(config)
    elif config.model_type == ModelType.ESMC_FITNESS_PREDICTOR:
        from models.esmc_fitness_predictor import ESMCFitnessPredictor
        model = ESMCFitnessPredictor(config)
    elif config.model_type == ModelType.VESPAG_FITNESS:
        from models.vespag_fitness_predictor import VespaGFitnessPredictor
        model = VespaGFitnessPredictor(config)
    elif config.model_type == ModelType.BLOSUM62:
        from models.blosum62_fitness_predictor import BLOSUM62FitnessPredictor
        model = BLOSUM62FitnessPredictor(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    return model.to(device)


def create_model_from_yaml(
    config_path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> BaseFitnessPredictor:
    """Create a model from a YAML config file.

    This is useful for models with pretrained weights (e.g., VespaG)
    that don't need a training checkpoint.

    Args:
        config_path: Path to YAML config file.
        device: Target device for the model.

    Returns:
        Initialized model instance.

    Example:
        >>> model = create_model_from_yaml("configs/vespag_fitness.yaml")
    """
    exp_config = ExperimentConfig.from_yaml(config_path)
    return create_model(exp_config.model, device)


def load_model(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> tuple[BaseFitnessPredictor, dict[str, Any]]:
    """Load a model from checkpoint or YAML config.

    Handles:
    - .pt/.pth checkpoint files (loads trained weights)
    - .yaml/.yml config files (creates model with pretrained weights if available)
    - Legacy checkpoint formats for backward compatibility

    Args:
        checkpoint_path: Path to checkpoint (.pt) or config (.yaml) file.
        device: Target device for the model.
        strict: Whether to strictly enforce state_dict key matching.

    Returns:
        Tuple of (model, checkpoint_dict) where checkpoint_dict contains
        all saved state including population genetics parameters.

    Example:
        >>> model, ckpt = load_model("trained_model.pt")
        >>> mu = torch.exp(ckpt["log_mu"])

        >>> # Or load from config (for pretrained models like VespaG)
        >>> model, ckpt = load_model("configs/vespag_fitness.yaml")
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    path = Path(checkpoint_path)

    # Handle YAML config files (for pretrained models)
    if path.suffix in ('.yaml', '.yml'):
        model = create_model_from_yaml(checkpoint_path, device)
        model.eval()
        # Return empty checkpoint dict for compatibility
        return model, {"config": model._config.to_dict()}

    # Handle checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle legacy checkpoint format
    if "config" not in checkpoint:
        config = _convert_legacy_checkpoint(checkpoint)
    else:
        config = ModelConfig.from_dict(checkpoint["config"])

    model = create_model(config, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.eval()

    return model, checkpoint


def _convert_legacy_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    """Convert legacy checkpoint format to new config format.

    Handles two legacy formats:
    - 'model_args' dict: from train.py's transformer path (vocab_size, seq_len, d_model, etc.)
    - Top-level keys (d_model, E, N): oldest format
    """
    if "model_args" in checkpoint:
        ma = checkpoint["model_args"]
        return ModelConfig(
            model_type=ModelType.FITNESS_PREDICTOR,
            seq_len=ma["seq_len"],
            alphabet=AlphabetConfig(
                amino_acids="ACDEFGHIKLMNPQRSTVWY*-",
            ),
            transformer=TransformerConfig(
                d_model=ma["d_model"],
                nhead=ma.get("nhead", 4),
                dim_feedforward=ma.get("dim_feedforward", 4 * ma["d_model"]),
                num_layers=ma.get("num_layers", 1),
                dropout=0.1,
            ),
        )

    # Oldest legacy format with top-level keys
    d_model = checkpoint.get("d_model", 128)
    E = checkpoint.get("E", 22)
    N = checkpoint.get("N", 3819)  # nucleotide length

    return ModelConfig(
        model_type=ModelType.FITNESS_PREDICTOR,
        seq_len=N // 3,
        alphabet=AlphabetConfig(
            nucleotides="ACGT-",
            amino_acids="ACDEFGHIKLMNPQRSTVWY*-",
        ),
        transformer=TransformerConfig(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            num_layers=1,
            dropout=0.1,
        ),
    )


def save_checkpoint(
    path: str,
    model: BaseFitnessPredictor,
    model_config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    **extra,
) -> None:
    """Save a training checkpoint.

    Args:
        path: Output path for the checkpoint file.
        model: The model to save.
        model_config: Model configuration.
        optimizer: Optimizer state to save.
        **extra: Additional data to include (epoch, loss, population params, etc).
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config.to_dict(),
        # Legacy keys for backward compatibility
        'd_model': model_config.transformer.d_model,
    }

    # Add extra data
    checkpoint.update(extra)

    # Handle experiment config separately if provided
    if 'experiment_config' in checkpoint:
        exp_config = checkpoint.pop('experiment_config')
        checkpoint['experiment_config'] = exp_config.to_dict()

    torch.save(checkpoint, path)
