"""Higher-capacity formula ranker and exact-evidence residual Transformer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LargeFormulaRankerConfig:
    max_peaks: int = 128
    max_mz: float = 500.0
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    fourier_frequencies: int = 8
    candidate_feature_dim: int = 25
    formula_hidden_dim: int = 512
    interaction_hidden_dim: int = 1024

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LargePeakEmbedding(nn.Module):
    def __init__(self, config: LargeFormulaRankerConfig) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(
            config.fourier_frequencies, dtype=torch.float32
        )
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.maximum_mz = float(config.max_mz)
        input_dim = 4 + 4 * config.fourier_frequencies
        self.projection = nn.Sequential(
            nn.Linear(input_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    def forward(self, peaks: Tensor, precursor_mz: Tensor) -> Tensor:
        mz = peaks[..., 0].clamp_min(0.0)
        intensity = peaks[..., 1].clamp_min(0.0).sqrt()
        neutral_loss = (precursor_mz[:, None] - mz).clamp_min(0.0)
        loss_valid = mz.le(precursor_mz[:, None]).to(peaks.dtype)
        mz_scaled = mz / self.maximum_mz
        loss_scaled = neutral_loss / self.maximum_mz
        mz_angle = torch.pi * mz_scaled[..., None] * self.frequencies
        loss_angle = torch.pi * loss_scaled[..., None] * self.frequencies
        features = torch.cat(
            (
                mz_scaled[..., None],
                intensity[..., None],
                loss_scaled[..., None],
                loss_valid[..., None],
                mz_angle.sin(),
                mz_angle.cos(),
                loss_angle.sin(),
                loss_angle.cos(),
            ),
            dim=-1,
        )
        return self.projection(features)


class LargeFormulaRanker(nn.Module):
    """Single-path base ranker without auxiliary element-count heads."""

    def __init__(self, config: LargeFormulaRankerConfig) -> None:
        super().__init__()
        self.config = config
        self.peak_embedding = LargePeakEmbedding(config)
        self.metadata_projection = nn.Sequential(
            nn.Linear(2, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spectrum_encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.spectrum_norm = nn.LayerNorm(config.d_model)
        self.formula_encoder = nn.Sequential(
            nn.LayerNorm(config.candidate_feature_dim),
            nn.Linear(config.candidate_feature_dim, config.formula_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.formula_hidden_dim, config.formula_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.formula_hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.spectrum_projection = nn.Linear(config.d_model, config.d_model)
        self.interaction_score = nn.Sequential(
            nn.LayerNorm(4 * config.d_model),
            nn.Linear(4 * config.d_model, config.interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.interaction_hidden_dim, 1),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_spectrum(
        self,
        peaks: Tensor,
        peak_padding_mask: Tensor,
        precursor_mz: Tensor,
        collision_energy: Tensor,
    ) -> Tensor:
        peak_tokens = self.peak_embedding(peaks, precursor_mz)
        metadata = torch.stack(
            (
                precursor_mz / self.config.max_mz,
                (collision_energy / 200.0).clamp(0.0, 2.0),
            ),
            dim=-1,
        )
        cls = self.metadata_projection(metadata)
        tokens = torch.cat((cls[:, None, :], peak_tokens), dim=1)
        cls_mask = torch.zeros(
            (len(peaks), 1), dtype=torch.bool, device=peaks.device
        )
        mask = torch.cat((cls_mask, peak_padding_mask), dim=1)
        encoded = self.spectrum_encoder(tokens, src_key_padding_mask=mask)
        return self.spectrum_norm(encoded[:, 0])

    def score_packed(
        self,
        spectrum_embeddings: Tensor,
        candidate_features: Tensor,
        candidate_owner: Tensor,
    ) -> Tensor:
        formula = self.formula_encoder(candidate_features)
        spectrum = self.spectrum_projection(spectrum_embeddings)[candidate_owner]
        normalized_spectrum = F.normalize(spectrum, dim=-1)
        normalized_formula = F.normalize(formula, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        dot = scale * (normalized_spectrum * normalized_formula).sum(dim=-1)
        interaction = torch.cat(
            (
                spectrum,
                formula,
                spectrum * formula,
                (spectrum - formula).abs(),
            ),
            dim=-1,
        )
        return dot + self.interaction_score(interaction).squeeze(-1)

    def score_batched(
        self,
        spectrum_embeddings: Tensor,
        candidate_features: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        batch, candidates, features = candidate_features.shape
        owner = torch.arange(batch, device=candidate_features.device).repeat_interleave(
            candidates
        )
        score = self.score_packed(
            spectrum_embeddings,
            candidate_features.reshape(batch * candidates, features),
            owner,
        ).reshape(batch, candidates)
        return score.masked_fill(candidate_mask, -1.0e4)


@dataclass(frozen=True)
class EvidenceTransformerConfig:
    base_spectrum_dim: int = 512
    candidate_feature_dim: int = 25
    evidence_feature_dim: int = 40
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 768
    dropout: float = 0.1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EvidenceResidualTransformer(nn.Module):
    """Candidate-specific evidence encoder with an identity-preserving output."""

    def __init__(self, config: EvidenceTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(config.evidence_feature_dim),
            nn.Linear(config.evidence_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.spectrum_projection = nn.Sequential(
            nn.LayerNorm(config.base_spectrum_dim),
            nn.Linear(config.base_spectrum_dim, config.d_model),
        )
        self.formula_projection = nn.Sequential(
            nn.LayerNorm(config.candidate_feature_dim),
            nn.Linear(config.candidate_feature_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(config.d_model)
        self.delta = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    @staticmethod
    def standardized_base(base_score: Tensor, candidate_mask: Tensor) -> Tensor:
        valid = (~candidate_mask).to(base_score.dtype)
        denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (base_score.masked_fill(candidate_mask, 0.0) * valid).sum(
            dim=1, keepdim=True
        ) / denominator
        variance = (
            (base_score - mean).masked_fill(candidate_mask, 0.0).square() * valid
        ).sum(dim=1, keepdim=True) / denominator
        return (base_score - mean) / variance.sqrt().clamp_min(1.0e-3)

    def forward(
        self,
        base_score: Tensor,
        spectrum_embedding: Tensor,
        candidate_features: Tensor,
        evidence_tokens: Tensor,
        evidence_padding_mask: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        base_z = self.standardized_base(base_score, candidate_mask)
        return self.forward_standardized(
            base_z,
            spectrum_embedding,
            candidate_features,
            evidence_tokens,
            evidence_padding_mask,
            candidate_mask,
        )

    def forward_standardized(
        self,
        standardized_base_score: Tensor,
        spectrum_embedding: Tensor,
        candidate_features: Tensor,
        evidence_tokens: Tensor,
        evidence_padding_mask: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        """Score candidates when base scores were normalized over the full domain."""

        batch, candidates, token_count, feature_dim = evidence_tokens.shape
        if standardized_base_score.shape != (batch, candidates):
            raise ValueError("standardized base score shape mismatch")
        if feature_dim != self.config.evidence_feature_dim:
            raise ValueError("unexpected evidence token dimension")
        if candidate_features.shape[:2] != (batch, candidates):
            raise ValueError("candidate feature shape mismatch")
        if evidence_padding_mask.shape != (batch, candidates, token_count):
            raise ValueError("evidence padding mask shape mismatch")
        if candidate_mask.shape != (batch, candidates):
            raise ValueError("candidate mask shape mismatch")

        spectrum_context = self.spectrum_projection(spectrum_embedding)
        spectrum_context = spectrum_context[:, None, :].expand(-1, candidates, -1)
        formula_context = self.formula_projection(candidate_features)
        cls = self.cls_token + spectrum_context + formula_context
        evidence = self.evidence_projection(evidence_tokens)
        sequence = torch.cat((cls[:, :, None, :], evidence), dim=2)
        flat_sequence = sequence.reshape(batch * candidates, token_count + 1, -1)
        cls_mask = torch.zeros(
            (batch, candidates, 1),
            dtype=torch.bool,
            device=standardized_base_score.device,
        )
        sequence_mask = torch.cat((cls_mask, evidence_padding_mask), dim=2)
        encoded = self.encoder(
            flat_sequence,
            src_key_padding_mask=sequence_mask.reshape(
                batch * candidates, token_count + 1
            ),
        )
        pooled = self.output_norm(encoded[:, 0]).reshape(batch, candidates, -1)
        delta = self.delta(
            torch.cat((pooled, standardized_base_score[..., None]), dim=-1)
        ).squeeze(-1)
        return (standardized_base_score + delta).masked_fill(
            candidate_mask, -1.0e4
        )
