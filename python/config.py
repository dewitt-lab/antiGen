"""Configuration dataclasses for fitness prediction models."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ModelType(str, Enum):
    """Supported model architectures."""
    FITNESS_PREDICTOR = "fitness_predictor"
    ESM2_FITNESS_PREDICTOR = "esm2_fitness_predictor"
    ESM2_POTTS_FITNESS = "esm2_potts_fitness"  # Conservation + Potts couplings
    VESPAG_FITNESS = "vespag_fitness"  # VespaG: ESM2-3B embeddings + FNN


class TrainingMode(str, Enum):
    """Training paradigms for fitness prediction."""
    PHYLODYNAMIC = "phylodynamic"      # Current approach: learn from birth-death dynamics
    PER_MUTATION = "per_mutation"      # New approach: treat each unique mutation as one example


@dataclass(frozen=True)
class AlphabetConfig:
    """Configuration for amino acid and nucleotide alphabets.

    Attributes:
        nucleotides: Allowed nucleotide characters.
        amino_acids: Allowed amino acid characters including special tokens.
    """
    nucleotides: str = "ACGT-"
    amino_acids: str = "ACDEFGHIKLMNPQRSTVWY*-"

    @property
    def vocab_size(self) -> int:
        """Number of amino acid tokens (E in the codebase)."""
        return len(self.amino_acids)

    @property
    def nuc_vocab_size(self) -> int:
        """Number of nucleotide tokens (D in the codebase)."""
        return len(self.nucleotides)


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the from-scratch transformer model.

    Attributes:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: Feedforward network dimension.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability.
    """
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 512
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )


@dataclass(frozen=True)
class ESM2Config:
    """Configuration for ESM-2 based model.

    Attributes:
        model_name: HuggingFace model identifier.
        hidden_size: ESM-2 hidden dimension (determined by model).
        freeze_backbone: Whether to freeze ESM-2 weights during training.
    """
    model_name: str = "facebook/esm2_t12_35M_UR50D"
    hidden_size: int = 480  # Fixed by the 35M model
    freeze_backbone: bool = True


@dataclass(frozen=True)
class VespaGConfig:
    """Configuration for VespaG-based model.

    VespaG uses ESM2 3B embeddings fed through a lightweight feedforward network.
    The ESM2 model is fixed to esm2_t36_3B_UR50D as this is what VespaG was designed with.

    Attributes:
        hidden_sizes: FNN hidden layer sizes. Default [256] matches VespaG paper.
        dropout: Dropout rate. Default 0.2 matches VespaG paper.
        freeze_backbone: Whether to freeze ESM2 backbone during training.
        normalize_scores: Apply sigmoid to normalize outputs to [0, 1].
        load_pretrained: Load pretrained VespaG FNN weights (requires vespag package).
    """
    hidden_sizes: tuple[int, ...] = (256,)  # VespaG default: single 256-dim hidden layer
    dropout: float = 0.2  # VespaG default dropout
    freeze_backbone: bool = True
    normalize_scores: bool = False  # Sigmoid to [0,1]
    load_pretrained: bool = True  # Load pretrained VespaG weights
    # Note: Uses ESM2 3B (esm2_t36_3B_UR50D) as VespaG was designed - not configurable


@dataclass(frozen=True)
class ModelConfig:
    """Unified model configuration.

    Attributes:
        model_type: Which model architecture to use.
        seq_len: Expected sequence length (in amino acids, N/3 in codebase).
        alphabet: Alphabet configuration.
        transformer: Config for from-scratch transformer (used if model_type is FITNESS_PREDICTOR).
        esm2: Config for ESM-2 model (used if model_type is ESM2_FITNESS_PREDICTOR).
    """
    model_type: ModelType = ModelType.FITNESS_PREDICTOR
    seq_len: int = 1273  # Default: (25381 - 21562) / 3 for spike protein
    alphabet: AlphabetConfig = field(default_factory=AlphabetConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    esm2: ESM2Config = field(default_factory=ESM2Config)
    vespag: VespaGConfig = field(default_factory=VespaGConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for checkpoint storage."""
        return {
            "model_type": self.model_type.value,
            "seq_len": self.seq_len,
            "alphabet": {
                "nucleotides": self.alphabet.nucleotides,
                "amino_acids": self.alphabet.amino_acids,
            },
            "transformer": {
                "d_model": self.transformer.d_model,
                "nhead": self.transformer.nhead,
                "dim_feedforward": self.transformer.dim_feedforward,
                "num_layers": self.transformer.num_layers,
                "dropout": self.transformer.dropout,
            },
            "esm2": {
                "model_name": self.esm2.model_name,
                "hidden_size": self.esm2.hidden_size,
                "freeze_backbone": self.esm2.freeze_backbone,
            },
            "vespag": {
                "hidden_sizes": list(self.vespag.hidden_sizes),
                "dropout": self.vespag.dropout,
                "freeze_backbone": self.vespag.freeze_backbone,
                "normalize_scores": self.vespag.normalize_scores,
                "load_pretrained": self.vespag.load_pretrained,
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        """Deserialize config from dictionary."""
        # Filter ESM2 config to only known fields (backward compat)
        esm2_fields = {'model_name', 'hidden_size', 'freeze_backbone'}
        esm2_dict = {k: v for k, v in d["esm2"].items() if k in esm2_fields}

        # Parse VespaG config (with backward compat for old checkpoints)
        vespag_dict = d.get("vespag", {})
        hidden_sizes = vespag_dict.get("hidden_sizes", [256])  # Default to VespaG architecture
        vespag_config = VespaGConfig(
            hidden_sizes=tuple(hidden_sizes),
            dropout=vespag_dict.get("dropout", 0.2),
            freeze_backbone=vespag_dict.get("freeze_backbone", True),
            normalize_scores=vespag_dict.get("normalize_scores", False),
            load_pretrained=vespag_dict.get("load_pretrained", True),
        )

        return cls(
            model_type=ModelType(d["model_type"]),
            seq_len=d["seq_len"],
            alphabet=AlphabetConfig(**d["alphabet"]),
            transformer=TransformerConfig(**d["transformer"]),
            esm2=ESM2Config(**esm2_dict),
            vespag=vespag_config,
        )


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 0.005
    warmup_steps: int = 0
    save_every: int = 0  # Save checkpoint every N batches (0 = only at end)
    seed: int = 42

    # Training mode: phylodynamic (default) or per_mutation (new approach)
    training_mode: TrainingMode = TrainingMode.PHYLODYNAMIC

    # Fitness-weighted mutation loss (phylodynamic mode only):
    # treats observed mutations as "selected" from alternatives based on fitness
    mutation_fitness_loss: bool = False  # Enable fitness-dependent mutation likelihood
    mutation_fitness_weight: float = 1.0  # Weight of this loss term relative to main loss

    # Per-mutation mode specific settings
    mutation_success_metric: str = "descendants"  # "descendants" or "time_on_tree"


@dataclass
class DataConfig:
    """Configuration for data processing."""
    start_pos: int = 21562
    end_pos: int = 25381
    dt: float = 0.1


@dataclass
class ExperimentConfig:
    """Full experiment configuration combining model, training, and data."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model": self.model.to_dict(),
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "warmup_steps": self.training.warmup_steps,
                "save_every": self.training.save_every,
                "seed": self.training.seed,
                "training_mode": self.training.training_mode.value,
                "mutation_fitness_loss": self.training.mutation_fitness_loss,
                "mutation_fitness_weight": self.training.mutation_fitness_weight,
                "mutation_success_metric": self.training.mutation_success_metric,
            },
            "data": {
                "start_pos": self.data.start_pos,
                "end_pos": self.data.end_pos,
                "dt": self.data.dt,
            },
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """Deserialize from dictionary."""
        # Handle training_mode enum conversion
        training_d = d["training"].copy()
        if "training_mode" in training_d:
            training_d["training_mode"] = TrainingMode(training_d["training_mode"])
        return cls(
            model=ModelConfig.from_dict(d["model"]),
            training=TrainingConfig(**training_d),
            data=DataConfig(**d["data"]),
            output_dir=d.get("output_dir", "checkpoints"),
        )

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)

        # Parse model config from YAML structure
        model_d = d.get("model", {})
        model_type_str = model_d.get("type", "transformer")

        if model_type_str == "esm2":
            model_type = ModelType.ESM2_FITNESS_PREDICTOR
            esm2_d = model_d.get("esm2", {})
            esm2_config = ESM2Config(
                model_name=esm2_d.get("model_name", "facebook/esm2_t12_35M_UR50D"),
                freeze_backbone=esm2_d.get("freeze_backbone", True),
            )
            transformer_config = TransformerConfig()
            vespag_config = VespaGConfig()
        elif model_type_str == "esm2_potts_fitness":
            model_type = ModelType.ESM2_POTTS_FITNESS
            esm2_d = model_d.get("esm2", {})
            esm2_config = ESM2Config(
                model_name=esm2_d.get("model_name", "facebook/esm2_t12_35M_UR50D"),
                freeze_backbone=esm2_d.get("freeze_backbone", True),
            )
            transformer_config = TransformerConfig()
            vespag_config = VespaGConfig()
        elif model_type_str == "vespag":
            model_type = ModelType.VESPAG_FITNESS
            vespag_d = model_d.get("vespag", {})
            hidden_sizes = vespag_d.get("hidden_sizes", [256])  # VespaG default
            vespag_config = VespaGConfig(
                hidden_sizes=tuple(hidden_sizes),
                dropout=vespag_d.get("dropout", 0.2),
                freeze_backbone=vespag_d.get("freeze_backbone", True),
                normalize_scores=vespag_d.get("normalize_scores", False),
                load_pretrained=vespag_d.get("load_pretrained", True),
            )
            esm2_config = ESM2Config()
            transformer_config = TransformerConfig()
        else:
            model_type = ModelType.FITNESS_PREDICTOR
            transformer_d = model_d.get("transformer", {})
            transformer_config = TransformerConfig(
                d_model=transformer_d.get("d_model", 128),
                nhead=transformer_d.get("nhead", 4),
                dim_feedforward=transformer_d.get("dim_feedforward", 512),
                num_layers=transformer_d.get("num_layers", 1),
                dropout=transformer_d.get("dropout", 0.1),
            )
            esm2_config = ESM2Config()
            vespag_config = VespaGConfig()

        # seq_len will be computed from data config
        data_d = d.get("data", {})
        start_pos = data_d.get("start_pos", 21562)
        end_pos = data_d.get("end_pos", 25381)
        seq_len = (end_pos - start_pos) // 3

        model_config = ModelConfig(
            model_type=model_type,
            seq_len=seq_len,
            transformer=transformer_config,
            esm2=esm2_config,
            vespag=vespag_config,
        )

        training_d = d.get("training", {})
        training_mode_str = training_d.get("training_mode", "phylodynamic")
        training_config = TrainingConfig(
            batch_size=training_d.get("batch_size", 64),
            epochs=training_d.get("epochs", 1),
            learning_rate=training_d.get("learning_rate", 0.005),
            warmup_steps=training_d.get("warmup_steps", 0),
            save_every=training_d.get("save_every", 0),
            seed=training_d.get("seed", 42),
            training_mode=TrainingMode(training_mode_str),
            mutation_fitness_loss=training_d.get("mutation_fitness_loss", False),
            mutation_fitness_weight=training_d.get("mutation_fitness_weight", 1.0),
            mutation_success_metric=training_d.get("mutation_success_metric", "descendants"),
        )

        data_config = DataConfig(
            start_pos=start_pos,
            end_pos=end_pos,
            dt=data_d.get("dt", 0.01),
        )

        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            output_dir=d.get("output_dir", "checkpoints"),
        )

    def save_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml

        # Convert to YAML-friendly structure
        if self.model.model_type == ModelType.ESM2_FITNESS_PREDICTOR:
            model_type_str = "esm2"
        elif self.model.model_type == ModelType.ESM2_POTTS_FITNESS:
            model_type_str = "esm2_potts_fitness"
        elif self.model.model_type == ModelType.VESPAG_FITNESS:
            model_type_str = "vespag"
        else:
            model_type_str = "transformer"

        d = {
            "model": {
                "type": model_type_str,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "epochs": self.training.epochs,
                "learning_rate": self.training.learning_rate,
                "warmup_steps": self.training.warmup_steps,
                "save_every": self.training.save_every,
                "seed": self.training.seed,
                "training_mode": self.training.training_mode.value,
                "mutation_fitness_loss": self.training.mutation_fitness_loss,
                "mutation_fitness_weight": self.training.mutation_fitness_weight,
                "mutation_success_metric": self.training.mutation_success_metric,
            },
            "data": {
                "start_pos": self.data.start_pos,
                "end_pos": self.data.end_pos,
                "dt": self.data.dt,
            },
            "output_dir": self.output_dir,
        }

        if model_type_str == "esm2":
            d["model"]["esm2"] = {
                "model_name": self.model.esm2.model_name,
                "freeze_backbone": self.model.esm2.freeze_backbone,
            }
        elif model_type_str == "vespag":
            d["model"]["vespag"] = {
                "hidden_sizes": list(self.model.vespag.hidden_sizes),
                "dropout": self.model.vespag.dropout,
                "freeze_backbone": self.model.vespag.freeze_backbone,
                "normalize_scores": self.model.vespag.normalize_scores,
                "load_pretrained": self.model.vespag.load_pretrained,
            }
        else:
            d["model"]["transformer"] = {
                "d_model": self.model.transformer.d_model,
                "nhead": self.model.transformer.nhead,
                "dim_feedforward": self.model.transformer.dim_feedforward,
                "num_layers": self.model.transformer.num_layers,
                "dropout": self.model.transformer.dropout,
            }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)
