"""VespaG-based fitness predictor using ESM2-3B embeddings + FNN."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from config import ModelConfig


class VespaGFNN(nn.Module):
    """Feedforward network for VespaG fitness prediction.

    Takes ESM2-3B embeddings (dim=2560) and outputs scores for 20 standard amino acids.

    Architecture:
        - Input: 2560 (ESM2-3B embedding dimension)
        - Hidden layers: configurable (default: [256] as in VespaG paper)
        - Output: 20 (standard amino acids)
        - Activation: LeakyReLU between layers
        - Dropout: Applied after each layer (default 0.2)
    """

    ESM2_3B_DIM = 2560  # Fixed embedding dimension for ESM2-3B
    NUM_STANDARD_AA = 20  # VespaG outputs 20 standard amino acids

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (256,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = self.ESM2_3B_DIM

        # Build hidden layers
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, self.NUM_STANDARD_AA))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

        # Initialize weights using Kaiming normal (as VespaG does)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming normal and biases to zero."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNN.

        Args:
            embeddings: (batch, seq_len, 2560) ESM2 embeddings

        Returns:
            (batch, seq_len, 20) fitness scores for standard amino acids
        """
        return self.net(embeddings)


class VespaGFitnessPredictor(BaseFitnessPredictor):
    """Fitness predictor using VespaG architecture: ESM2-3B embeddings + FNN.

    Architecture:
        1. Input: Integer AA indices using internal alphabet (22 tokens)
        2. Alphabet mapping: Convert to ESM-2 token indices
        3. ESM2-3B forward: Extract per-residue embeddings (2560 dim)
        4. FNN: Project to 20 standard AA fitness scores
        5. Output mapping: Map to internal 22-token alphabet (add -inf for *, -)

    For sequences > 1022 tokens, uses sliding window with 2 forward passes.
    """

    ESM2_MAX_LEN = 1022  # Max tokens excluding <cls> and <eos>
    ESM2_3B_MODEL = "facebook/esm2_t36_3B_UR50D"  # Fixed model for VespaG
    ESM2_3B_DIM = 2560

    # VespaG uses 20 standard amino acids in this order
    VESPAG_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    # Score for stop codon and gap (large negative, numerically stable)
    SPECIAL_TOKEN_SCORE = -100.0

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        vespag_config = config.vespag

        # Lazy import to avoid dependency when not using ESM-2
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for VespaG. "
                "Install with: pip install transformers"
            )

        # Load ESM2-3B backbone (embeddings only, no LM head)
        import logging
        _logger = logging.getLogger("transformers.modeling_utils")
        _prev_level = _logger.level
        _logger.setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(self.ESM2_3B_MODEL)
        self.esm = AutoModel.from_pretrained(self.ESM2_3B_MODEL)

        _logger.setLevel(_prev_level)

        # Build alphabet mappings
        self._build_alphabet_mapping()

        # Create FNN head with proper architecture
        self.fnn = VespaGFNN(
            hidden_sizes=vespag_config.hidden_sizes,
            dropout=vespag_config.dropout,
        )

        # Load pretrained VespaG FNN weights if requested
        if vespag_config.load_pretrained:
            self._load_pretrained_fnn()

        # Store config options
        self.normalize_scores = vespag_config.normalize_scores

        # Freeze backbone if configured
        if vespag_config.freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False

    def _load_pretrained_fnn(self) -> None:
        """Load pretrained VespaG FNN weights.

        Attempts to load weights from:
        1. The vespag package if installed
        2. A cached download location

        Raises:
            ImportError: If vespag package is not installed and weights not found.
        """
        import logging
        logger = logging.getLogger(__name__)

        # Try loading from vespag package
        try:
            from vespag.utils import load_model, DEFAULT_MODEL_PARAMETERS
            from vespag.models.fnn import FNN

            # Load the pretrained VespaG model
            params = DEFAULT_MODEL_PARAMETERS.copy()
            vespag_model = load_model(**params)

            # Extract state dict and load directly (we use same naming as VespaG)
            vespag_state = vespag_model.state_dict()
            self.fnn.load_state_dict(vespag_state, strict=True)
            logger.info("Loaded pretrained VespaG FNN weights from vespag package")

        except ImportError:
            logger.warning(
                "vespag package not installed. Install with: pip install vespag\n"
                "Using randomly initialized FNN weights instead."
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained VespaG weights: {e}")

    def _build_alphabet_mapping(self) -> None:
        """Build bidirectional alphabet mappings.

        Internal alphabet: "ACDEFGHIKLMNPQRSTVWY*-" (22 tokens)
        VespaG alphabet: "ACDEFGHIKLMNPQRSTVWY" (20 tokens)

        Creates:
            - alphabet_map: internal index -> ESM-2 token id
            - vespag_to_internal: VespaG output index -> internal alphabet index
        """
        internal_alphabet = self._config.alphabet.amino_acids

        # Internal -> ESM-2 token ids
        to_esm = []
        for char in internal_alphabet:
            if char == '*':
                to_esm.append(self.tokenizer.unk_token_id)
            elif char == '-':
                # Map gap to <pad> or <unk>
                to_esm.append(self.tokenizer.pad_token_id or self.tokenizer.unk_token_id)
            else:
                to_esm.append(self.tokenizer.convert_tokens_to_ids(char))

        self.register_buffer('alphabet_map', torch.tensor(to_esm, dtype=torch.long))

        # VespaG output (20 AAs) -> internal alphabet (22 tokens)
        # Build mapping: for each VespaG position, which internal index?
        vespag_to_internal = []
        for char in self.VESPAG_ALPHABET:
            internal_idx = internal_alphabet.index(char)
            vespag_to_internal.append(internal_idx)

        self.register_buffer(
            'vespag_to_internal',
            torch.tensor(vespag_to_internal, dtype=torch.long)
        )

        # Store special token indices
        self.stop_codon_idx = internal_alphabet.index('*')
        self.gap_idx = internal_alphabet.index('-')

    def _esm_forward(self, esm_tokens: torch.Tensor) -> torch.Tensor:
        """Single ESM-2 forward pass returning embeddings.

        Args:
            esm_tokens: (batch, chunk_len) ESM-2 token ids

        Returns:
            (batch, chunk_len, 2560) per-residue embeddings
        """
        batch_size = esm_tokens.shape[0]
        device = esm_tokens.device

        # Add CLS and EOS tokens
        cls_tokens = torch.full(
            (batch_size, 1), self.tokenizer.cls_token_id,
            dtype=torch.long, device=device
        )
        eos_tokens = torch.full(
            (batch_size, 1), self.tokenizer.eos_token_id,
            dtype=torch.long, device=device
        )
        esm_input = torch.cat([cls_tokens, esm_tokens, eos_tokens], dim=1)

        # Create attention mask
        attention_mask = torch.ones_like(esm_input)

        # Forward through ESM-2 (embeddings only)
        outputs = self.esm(
            input_ids=esm_input,
            attention_mask=attention_mask,
        )

        # Exclude CLS and EOS from hidden states
        return outputs.last_hidden_state[:, 1:-1, :]

    def _map_to_internal_alphabet(self, vespag_scores: torch.Tensor) -> torch.Tensor:
        """Map VespaG 20-AA scores to internal 22-token alphabet.

        Args:
            vespag_scores: (batch, seq_len, 20) scores for standard amino acids

        Returns:
            (batch, seq_len, 22) scores mapped to internal alphabet
        """
        batch_size, seq_len, _ = vespag_scores.shape
        device = vespag_scores.device
        internal_vocab_size = self._config.alphabet.vocab_size  # 22

        # Create output tensor with special token scores
        output = torch.full(
            (batch_size, seq_len, internal_vocab_size),
            self.SPECIAL_TOKEN_SCORE,
            device=device,
            dtype=vespag_scores.dtype
        )

        # Map VespaG scores to corresponding internal positions
        # vespag_to_internal: (20,) maps VespaG index -> internal index
        output[:, :, self.vespag_to_internal] = vespag_scores

        return output

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VespaG model.

        For sequences > 1022, uses sliding window with 2 forward passes:
        - Pass 1: positions 0 to ESM2_MAX_LEN-1
        - Pass 2: positions (seq_len - ESM2_MAX_LEN) to seq_len-1
        - Overlap region: averaged

        Args:
            x: Integer indices tensor of shape (batch, seq_len) using
               internal alphabet (ACDEFGHIKLMNPQRSTVWY*-).

        Returns:
            Fitness scores of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Map from internal alphabet to ESM-2 tokens
        esm_tokens = self.alphabet_map[x]  # (batch, seq_len)

        if seq_len <= self.ESM2_MAX_LEN:
            # Short sequence - single forward pass
            embeddings = self._esm_forward(esm_tokens)  # (batch, seq_len, 2560)
            vespag_scores = self.fnn(embeddings)  # (batch, seq_len, 20)
        else:
            # Long sequence - sliding window with 2 passes
            split_point = seq_len - self.ESM2_MAX_LEN  # Start of second chunk

            chunk1 = esm_tokens[:, :self.ESM2_MAX_LEN]  # 0 to MAX_LEN-1
            chunk2 = esm_tokens[:, split_point:]        # split_point to end

            emb1 = self._esm_forward(chunk1)  # (batch, MAX_LEN, 2560)
            emb2 = self._esm_forward(chunk2)  # (batch, MAX_LEN, 2560)

            scores1 = self.fnn(emb1)  # (batch, MAX_LEN, 20)
            scores2 = self.fnn(emb2)  # (batch, MAX_LEN, 20)

            # Stitch together with averaging in overlap
            vespag_scores = torch.zeros(batch_size, seq_len, 20, device=device)

            # Positions 0 to split_point-1: only from pass 1
            vespag_scores[:, :split_point] = scores1[:, :split_point]

            # Overlap region: split_point to MAX_LEN-1 (average both)
            overlap_len = self.ESM2_MAX_LEN - split_point
            vespag_scores[:, split_point:self.ESM2_MAX_LEN] = (
                scores1[:, split_point:] + scores2[:, :overlap_len]
            ) / 2

            # Positions MAX_LEN to end: only from pass 2
            vespag_scores[:, self.ESM2_MAX_LEN:] = scores2[:, overlap_len:]

        # Optional sigmoid normalization
        if self.normalize_scores:
            vespag_scores = torch.sigmoid(vespag_scores)

        # Map to internal 22-token alphabet
        fitness = self._map_to_internal_alphabet(vespag_scores)

        return fitness
