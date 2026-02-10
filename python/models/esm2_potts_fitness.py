"""ESM-2 Potts-style fitness predictor with conservation + coupling decomposition.

Decomposes fitness into:
1. Conservation (h_i): Per-position bias from ESM-2 embeddings
2. Potts couplings (J_ij): Pairwise interactions from ESM-2 attention weights

fitness(seq) = sum_i h_i(a_i) + sum_{i<j} J_ij(a_i, a_j)

Inspired by esMRF (Pottsformer).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from config import ModelConfig


class MLP(nn.Module):
    """Simple MLP for projecting attention to couplings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.output_layer(x)


class ESM2PottsFitnessPredictor(BaseFitnessPredictor):
    """Fitness predictor decomposing into conservation + Potts couplings.

    Architecture:
        1. ESM-2 forward pass with attention weights
        2. Conservation (h_i): Linear projection of ESM embeddings -> (L, 22)
        3. Couplings (J_ij): Attention weights -> MLP -> (L, L, 22, 22)
        4. Final fitness: h_i(a) + sum_j J_ij(a, seq[j])

    This learns:
        - Single-site fitness effects (conservation)
        - Pairwise epistatic interactions (couplings)
    """

    def __init__(
        self,
        config: ModelConfig,
        attn_layers: Optional[List[int]] = None,
        coupling_hidden_dim: int = 256,
        coupling_mlp_layers: int = 1,
        coupling_dropout: float = 0.0,
        symmetrize_couplings: bool = True,
        use_couplings: bool = True,
    ) -> None:
        """Initialize ESM2 Potts fitness predictor.

        Args:
            config: Model configuration.
            attn_layers: Which ESM-2 layers to extract attention from.
                         Default: last layer only.
            coupling_hidden_dim: Hidden dimension for coupling MLP.
            coupling_mlp_layers: Number of MLP layers for coupling projection.
            coupling_dropout: Dropout in coupling MLP.
            symmetrize_couplings: Whether to symmetrize J_ij = J_ji.
            use_couplings: Whether to use pairwise couplings (False = conservation only).
        """
        super().__init__(config)

        self.symmetrize_couplings = symmetrize_couplings
        self.use_couplings = use_couplings

        esm_config = config.esm2
        model_name = esm_config.model_name

        # Load ESM-2
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("transformers library required for ESM-2")

        import logging
        _logger = logging.getLogger("transformers.modeling_utils")
        _prev_level = _logger.level
        _logger.setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.esm = AutoModel.from_pretrained(model_name, output_attentions=True)

        _logger.setLevel(_prev_level)

        # Get model properties from loaded config (more reliable than hardcoded)
        self.num_layers = self.esm.config.num_hidden_layers
        self.num_heads = self.esm.config.num_attention_heads
        self.hidden_dim = self.esm.config.hidden_size

        # Which layers to use for attention
        if attn_layers is None:
            attn_layers = [self.num_layers - 1]  # Last layer by default
        self.attn_layers = attn_layers

        # Freeze ESM backbone if configured
        if esm_config.freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False

        # Internal vocab size (our alphabet)
        self.internal_vocab = config.alphabet.vocab_size  # 22

        # Build alphabet mapping
        self._build_alphabet_mapping()

        # Conservation head: ESM embeddings -> per-position scalar
        # h_i = single value per position (column-level fitness/tolerance)
        # This captures "how important/conserved is this position" not "which AA is good"
        self.conservation_head = nn.Linear(self.hidden_dim, 1)

        # Coupling head: attention -> pairwise couplings
        # J_ij(a, b) for each position pair and amino acid pair
        if self.use_couplings:
            # Input: concatenated attention from all specified layers
            # Shape per layer: (batch, heads, L, L)
            attn_input_dim = len(self.attn_layers) * self.num_heads
            coupling_output_dim = self.internal_vocab * self.internal_vocab  # 22 * 22 = 484

            if coupling_mlp_layers > 1:
                self.coupling_head = MLP(
                    input_dim=attn_input_dim,
                    hidden_dim=coupling_hidden_dim,
                    output_dim=coupling_output_dim,
                    num_layers=coupling_mlp_layers,
                    dropout=coupling_dropout,
                )
            else:
                self.coupling_head = nn.Linear(attn_input_dim, coupling_output_dim)

            # Initialize coupling weights small
            for param in self.coupling_head.parameters():
                nn.init.normal_(param, mean=0, std=0.01)

        # Weighting between conservation and coupling terms
        self.coupling_weight = nn.Parameter(torch.tensor(1.0))

    def _build_alphabet_mapping(self) -> None:
        """Build mapping between internal alphabet and ESM-2 tokens."""
        internal_alphabet = self._config.alphabet.amino_acids

        # Internal -> ESM-2
        to_esm = []
        for char in internal_alphabet:
            if char == "*":
                to_esm.append(self.tokenizer.unk_token_id)
            else:
                to_esm.append(self.tokenizer.convert_tokens_to_ids(char))
        self.register_buffer("alphabet_map", torch.tensor(to_esm, dtype=torch.long))

        # ESM-2 -> Internal
        esm_vocab_size = self.esm.config.vocab_size
        from_esm = torch.full((esm_vocab_size,), -1, dtype=torch.long)
        for internal_idx, char in enumerate(internal_alphabet):
            if char == "*":
                esm_idx = self.tokenizer.unk_token_id
            else:
                esm_idx = self.tokenizer.convert_tokens_to_ids(char)
            from_esm[esm_idx] = internal_idx
        self.register_buffer("esm_to_internal", from_esm)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing conservation + coupling fitness.

        Args:
            x: (batch, seq_len) tensor of internal alphabet indices.

        Returns:
            (batch, seq_len, vocab_size) fitness scores for each position and AA.
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Map to ESM-2 tokens
        esm_tokens = self.alphabet_map[x]

        # Add CLS and EOS
        cls_tokens = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
        eos_tokens = torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=device)
        esm_input = torch.cat([cls_tokens, esm_tokens, eos_tokens], dim=1)

        # Forward through ESM-2
        outputs = self.esm(
            input_ids=esm_input,
            attention_mask=torch.ones_like(esm_input),
            output_attentions=True,
            output_hidden_states=True,
        )

        # Get embeddings from last layer (exclude CLS and EOS)
        # hidden_states is tuple of (layer0, layer1, ..., layerN)
        embeddings = outputs.hidden_states[-1][:, 1:-1, :]  # (batch, seq_len, hidden)

        # Conservation term: h_i scalar per position (column-level tolerance)
        conservation_scalar = self.conservation_head(embeddings)  # (batch, seq_len, 1)
        # Broadcast to all AAs: same value for every AA at this position
        conservation = conservation_scalar.expand(-1, -1, self.internal_vocab)  # (batch, seq_len, vocab)

        if not self.use_couplings:
            return conservation

        # Coupling term: extract attention from specified layers
        # attentions is tuple of (layer0_attn, layer1_attn, ...)
        # Each attention: (batch, heads, seq_len+2, seq_len+2)
        attn_list = []
        for layer_idx in self.attn_layers:
            attn = outputs.attentions[layer_idx][:, :, 1:-1, 1:-1]  # Remove CLS/EOS
            attn_list.append(attn)

        # Stack attentions: (batch, num_layers * heads, L, L)
        attentions = torch.cat(attn_list, dim=1)

        # Permute to (batch, L, L, num_layers * heads) for MLP
        attentions = attentions.permute(0, 2, 3, 1)

        # Project to couplings: (batch, L, L, vocab * vocab)
        couplings = self.coupling_head(attentions)

        # Reshape to (batch, L, L, vocab, vocab)
        couplings = couplings.view(batch_size, seq_len, seq_len, self.internal_vocab, self.internal_vocab)

        # Symmetrize if requested: J_ij(a,b) = J_ji(b,a)
        if self.symmetrize_couplings:
            couplings = (couplings + couplings.permute(0, 2, 1, 4, 3)) / 2.0

        # Compute coupling contribution for each position
        # For position j, we want sum_i J_ij(a, x_i) for all a
        # x is the current sequence: (batch, seq_len)
        # x_onehot: (batch, seq_len, vocab)
        x_onehot = F.one_hot(x, num_classes=self.internal_vocab).float()

        # Coupling contribution: sum over i of J_ij(a, x_i)
        # couplings: (batch, L_j, L_i, vocab_a, vocab_b) - J_ij(a, b)
        # x_onehot: (batch, L_i, vocab_b) - one-hot of sequence
        # Result: (batch, L_j, vocab_a) = sum_i J_ij(a, x_i)
        # Using n=batch, j=pos_j, i=pos_i, a=vocab_a, b=vocab_b
        coupling_contrib = torch.einsum("njiab,nib->nja", couplings, x_onehot)

        # Combine conservation and coupling
        fitness = conservation + self.coupling_weight * coupling_contrib

        return fitness

    def get_conservation_and_couplings(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get separate conservation and coupling terms for analysis.

        Args:
            x: (batch, seq_len) tensor of internal alphabet indices.

        Returns:
            Tuple of:
                - conservation: (batch, seq_len, vocab) per-position fitness
                - couplings: (batch, seq_len, seq_len, vocab, vocab) pairwise couplings
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Map to ESM-2 tokens
        esm_tokens = self.alphabet_map[x]

        # Add CLS and EOS
        cls_tokens = torch.full((batch_size, 1), self.tokenizer.cls_token_id, dtype=torch.long, device=device)
        eos_tokens = torch.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=torch.long, device=device)
        esm_input = torch.cat([cls_tokens, esm_tokens, eos_tokens], dim=1)

        # Forward through ESM-2
        outputs = self.esm(
            input_ids=esm_input,
            attention_mask=torch.ones_like(esm_input),
            output_attentions=True,
            output_hidden_states=True,
        )

        # Conservation (scalar per position)
        embeddings = outputs.hidden_states[-1][:, 1:-1, :]
        conservation_scalar = self.conservation_head(embeddings)  # (batch, seq_len, 1)
        # Broadcast to all AAs for consistency with output shape
        conservation = conservation_scalar.expand(-1, -1, self.internal_vocab)  # (batch, seq_len, vocab)

        # Couplings
        if not self.use_couplings:
            return conservation, None

        attn_list = []
        for layer_idx in self.attn_layers:
            attn = outputs.attentions[layer_idx][:, :, 1:-1, 1:-1]
            attn_list.append(attn)

        attentions = torch.cat(attn_list, dim=1).permute(0, 2, 3, 1)
        couplings = self.coupling_head(attentions)
        couplings = couplings.view(batch_size, seq_len, seq_len, self.internal_vocab, self.internal_vocab)

        if self.symmetrize_couplings:
            couplings = (couplings + couplings.permute(0, 2, 1, 4, 3)) / 2.0

        return conservation, couplings
