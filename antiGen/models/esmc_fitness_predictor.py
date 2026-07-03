"""ESM C based fitness predictor."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig


class ESMCFitnessPredictor(BaseFitnessPredictor):
    """Fitness predictor using ESM C architecture with sequence head.

    Architecture:
        1. Input: Integer AA indices using internal alphabet
        2. Alphabet mapping: Convert to ESM C token indices
        3. ESMC: Full pretrained model with sequence head
        4. Output mapping: Map ESM C logits back to internal alphabet

    Available models:
        - esmc_300m (30 layers)
        - esmc_600m (36 layers)
        - esmc_6b  (80 layers)
    """

    # ESM-C's vocab has '-' as a real learned token but treats '*' as <unk> (same
    # as ESM-2). Gap predictions are legitimate; stop is <unk> reused.
    OUTPUTS_LEGITIMATE_GAP = True
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        esmc_config = config.esmc

        # Lazy import to avoid dependency when not using ESM C
        try:
            from esm.models.esmc import ESMC
        except ImportError:
            raise ImportError(
                "esm library required for ESM C. "
                "Install with: pip install esm"
            )

        # Load pretrained ESM C model
        # from_pretrained handles downloading and returns an ESMC nn.Module
        self.esmc = ESMC.from_pretrained(esmc_config.model_name)

        # Build alphabet mappings (both directions)
        self._build_alphabet_mapping()

        # Freeze backbone if configured, but keep sequence_head trainable
        if esmc_config.freeze_backbone:
            # Freeze embedding and transformer
            for param in self.esmc.embed.parameters():
                param.requires_grad = False
            for param in self.esmc.transformer.parameters():
                param.requires_grad = False

            # Unfreeze last k transformer blocks if requested
            if esmc_config.finetune_layers > 0:
                blocks = self.esmc.transformer.blocks
                n_layers = len(blocks)
                k = min(esmc_config.finetune_layers, n_layers)
                for block in blocks[n_layers - k:]:
                    for param in block.parameters():
                        param.requires_grad = True

    def _build_alphabet_mapping(self) -> None:
        """Build bidirectional alphabet mappings.

        Internal alphabet: "ACDEFGHIKLMNPQRSTVWY*-" (22 tokens)

        Creates:
            - alphabet_map: internal index -> ESM C token id
            - esmc_to_internal: ESM C token id -> internal index
        """
        internal_alphabet = self._config.alphabet.amino_acids
        tokenizer = self.esmc.tokenizer

        # Internal -> ESM C
        to_esmc = []
        for char in internal_alphabet:
            if char == '*':
                # Map stop codon to the UNK token
                to_esmc.append(tokenizer.convert_tokens_to_ids('<unk>'))
            else:
                to_esmc.append(tokenizer.convert_tokens_to_ids(char))

        self.register_buffer('alphabet_map', torch.tensor(to_esmc, dtype=torch.long))

    def _map_logits_to_internal(self, esmc_logits: torch.Tensor) -> torch.Tensor:
        """Map ESM C logits to internal alphabet.

        Args:
            esmc_logits: (batch, seq_len, esmc_vocab_size) logits from sequence head

        Returns:
            (batch, seq_len, internal_vocab_size) logits mapped to internal alphabet
        """
        return esmc_logits[:, :, self.alphabet_map]

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ESM C model with sequence head.

        Args:
            x: Integer indices tensor of shape (batch, seq_len) using
               internal alphabet (ACDEFGHIKLMNPQRSTVWY*-).

        Returns:
            Fitness scores of shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x.shape
        device = x.device
        tokenizer = self.esmc.tokenizer

        # Map from internal alphabet to ESM C tokens
        esmc_tokens = self.alphabet_map[x]  # (batch, seq_len)

        # Add BOS and EOS tokens
        bos = torch.full(
            (batch_size, 1), tokenizer.cls_token_id,
            dtype=torch.long, device=device,
        )
        eos = torch.full(
            (batch_size, 1), tokenizer.eos_token_id,
            dtype=torch.long, device=device,
        )
        esmc_input = torch.cat([bos, esmc_tokens, eos], dim=1)

        # Forward through ESM C (returns ESMCOutput with sequence_logits)
        output = self.esmc(sequence_tokens=esmc_input)

        # Strip BOS/EOS from logits: (batch, seq_len+2, vocab) -> (batch, seq_len, vocab)
        esmc_logits = output.sequence_logits[:, 1:-1, :]

        # Map to internal alphabet
        logits = self._map_logits_to_internal(esmc_logits)

        return logits
