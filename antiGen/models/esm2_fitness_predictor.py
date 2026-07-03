"""ESM-2 based fitness predictor."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from antiGen.models.base import BaseFitnessPredictor

if TYPE_CHECKING:
    from antiGen.config import ModelConfig


class ESM2FitnessPredictor(BaseFitnessPredictor):
    """Fitness predictor using full ESM-2 architecture with LM head.

    Architecture:
        1. Input: Integer AA indices using internal alphabet
        2. Alphabet mapping: Convert to ESM-2 token indices
        3. EsmForMaskedLM: Full pretrained model with LM head
        4. Output mapping: Map ESM-2 logits back to internal alphabet

    Uses the exact ESM-2 architecture. The LM head outputs logits over ESM's
    vocabulary, which we map back to our internal alphabet.

    For sequences > 1022 tokens, uses sliding window with 2 forward passes.
    """

    ESM2_MAX_LEN = 1022  # Max tokens excluding <cls> and <eos>

    # ESM-2's vocab has '-' as a real learned token but treats '*' as <unk>. Gap
    # predictions are legitimate; stop predictions are just the <unk> logit reused,
    # so mask stop out of the "All" hit-rate but keep gap.
    OUTPUTS_LEGITIMATE_GAP = True
    OUTPUTS_LEGITIMATE_STOP = False

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        esm_config = config.esm2

        # Lazy import to avoid dependency when not using ESM-2
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
        except ImportError:
            raise ImportError(
                "transformers library required for ESM-2. "
                "Install with: pip install transformers"
            )

        # Load full pretrained ESM-2 with LM head using Auto classes
        # Suppress position_ids warning (checkpoint format mismatch with transformers 5.x,
        # but values are correct and model works fine)
        import logging
        _logger = logging.getLogger("transformers.modeling_utils")
        _prev_level = _logger.level
        _logger.setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(esm_config.model_name)
        self.esm = AutoModelForMaskedLM.from_pretrained(esm_config.model_name)

        _logger.setLevel(_prev_level)

        # Build alphabet mappings (both directions)
        self._build_alphabet_mapping()

        # Freeze backbone (encoder) if configured, but keep LM head trainable
        if esm_config.freeze_backbone:
            # Freeze entire encoder/embeddings (not the LM head)
            for param in self.esm.esm.parameters():
                param.requires_grad = False

            # Unfreeze last k encoder layers if requested
            if esm_config.finetune_layers > 0:
                encoder_layers = self.esm.esm.encoder.layer
                n_layers = len(encoder_layers)
                k = min(esm_config.finetune_layers, n_layers)
                for layer in encoder_layers[n_layers - k:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def _build_alphabet_mapping(self) -> None:
        """Build bidirectional alphabet mappings.

        Internal alphabet: "ACDEFGHIKLMNPQRSTVWY*-" (22 tokens)

        Creates:
            - alphabet_map: internal index -> ESM-2 token id
            - esm_to_internal: ESM-2 token id -> internal index (for output mapping)
        """
        internal_alphabet = self._config.alphabet.amino_acids
        vocab_size = len(internal_alphabet)

        # Internal -> ESM-2
        to_esm = []
        for char in internal_alphabet:
            if char == '*':
                to_esm.append(self.tokenizer.unk_token_id)
            else:
                to_esm.append(self.tokenizer.convert_tokens_to_ids(char))

        self.register_buffer('alphabet_map', torch.tensor(to_esm, dtype=torch.long))

        # ESM-2 -> Internal (for mapping logits back)
        # ESM vocab is ~33 tokens, map each to internal index (or -1 if not in our alphabet)
        esm_vocab_size = self.esm.config.vocab_size
        from_esm = torch.full((esm_vocab_size,), -1, dtype=torch.long)

        for internal_idx, char in enumerate(internal_alphabet):
            if char == '*':
                esm_idx = self.tokenizer.unk_token_id
            else:
                esm_idx = self.tokenizer.convert_tokens_to_ids(char)
            from_esm[esm_idx] = internal_idx

        self.register_buffer('esm_to_internal', from_esm)

    def _esm_forward(self, esm_tokens: torch.Tensor) -> torch.Tensor:
        """Single ESM-2 forward pass returning logits.

        Args:
            esm_tokens: (batch, chunk_len) ESM-2 token ids

        Returns:
            (batch, chunk_len, esm_vocab_size) logits from LM head
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

        # Forward through ESM-2 with LM head
        outputs = self.esm(
            input_ids=esm_input,
            attention_mask=attention_mask,
        )

        # Exclude CLS and EOS from logits
        return outputs.logits[:, 1:-1, :]

    def _map_logits_to_internal(self, esm_logits: torch.Tensor) -> torch.Tensor:
        """Map ESM-2 logits to internal alphabet.

        Args:
            esm_logits: (batch, seq_len, esm_vocab_size) logits from ESM LM head

        Returns:
            (batch, seq_len, internal_vocab_size) logits mapped to internal alphabet
        """
        # Gather logits for tokens that map to our internal alphabet
        # alphabet_map: (internal_vocab_size,) -> esm token ids
        # We want logits[:, :, alphabet_map] to get (batch, seq_len, internal_vocab_size)
        return esm_logits[:, :, self.alphabet_map]

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through full ESM-2 model with LM head.

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
        internal_vocab_size = self._config.alphabet.vocab_size

        # Map from internal alphabet to ESM-2 tokens
        esm_tokens = self.alphabet_map[x]  # (batch, seq_len)

        if seq_len <= self.ESM2_MAX_LEN:
            # Short sequence - single forward pass
            esm_logits = self._esm_forward(esm_tokens)
            fitness = self._map_logits_to_internal(esm_logits)
        else:
            # Long sequence - sliding window with 2 passes
            split_point = seq_len - self.ESM2_MAX_LEN  # Start of second chunk

            chunk1 = esm_tokens[:, :self.ESM2_MAX_LEN]  # 0 to MAX_LEN-1
            chunk2 = esm_tokens[:, split_point:]        # split_point to end

            logits1 = self._esm_forward(chunk1)  # (batch, MAX_LEN, esm_vocab)
            logits2 = self._esm_forward(chunk2)  # (batch, MAX_LEN, esm_vocab)

            # Map to internal alphabet
            logits1 = self._map_logits_to_internal(logits1)
            logits2 = self._map_logits_to_internal(logits2)

            # Stitch together with averaging in overlap
            fitness = torch.zeros(batch_size, seq_len, internal_vocab_size, device=device)

            # Positions 0 to split_point-1: only from pass 1
            fitness[:, :split_point] = logits1[:, :split_point]

            # Overlap region: split_point to MAX_LEN-1 (average both)
            overlap_len = self.ESM2_MAX_LEN - split_point
            fitness[:, split_point:self.ESM2_MAX_LEN] = (
                logits1[:, split_point:] + logits2[:, :overlap_len]
            ) / 2

            # Positions MAX_LEN to end: only from pass 2
            fitness[:, self.ESM2_MAX_LEN:] = logits2[:, overlap_len:]

        return fitness
