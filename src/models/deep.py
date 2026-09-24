
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- data
@dataclass
class Bundle:
    demand: np.ndarray        # [T, N, 2] int16
    calendar: np.ndarray      # [T, C] float32
    weather: np.ndarray       # [T, W] float32
    edges: np.ndarray         # [2, E] int32
    static: np.ndarray        # [N, S] float32
    dst_mask: np.ndarray      # [T] bool
    meta: dict

    @property
    def n_steps(self) -> int:
        return self.demand.shape[0]

    @property
    def n_regions(self) -> int:
        return self.demand.shape[1]

    @classmethod
    def load(cls, path: str | Path) -> "Bundle":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        return cls(
            demand=np.load(path / "demand.npy"),
            calendar=np.load(path / "calendar.npy"),
            weather=np.load(path / "weather.npy"),
            edges=np.load(path / "edges.npy"),
            static=np.load(path / "node_static.npy"),
            dst_mask=np.load(path / "dst_mask.npy"),
            meta=meta,
        )

    def scaled_demand(self) -> np.ndarray:
        """z-scored with TRAIN-fitted statistics carried in the bundle."""
        mean = np.asarray(self.meta["scaling"]["demand_mean"], dtype=np.float32)
        std = np.asarray(self.meta["scaling"]["demand_std"], dtype=np.float32)
        return (self.demand.astype(np.float32) - mean) / std

    def scaled_weather(self) -> np.ndarray:
        mean = np.asarray(self.meta["scaling"]["weather_mean"], dtype=np.float32)
        std = np.asarray(self.meta["scaling"]["weather_std"], dtype=np.float32)
        return (self.weather - mean) / std

    def scaled_static(self) -> np.ndarray:
        mean = np.asarray(self.meta["scaling"]["static_mean"], dtype=np.float32)
        std = np.asarray(self.meta["scaling"]["static_std"], dtype=np.float32)
        return (self.static - mean) / std


def valid_window_starts(bundle: Bundle, split: str, window: int, seasonal_max: int) -> np.ndarray:
    """Indices `t` where a full history window AND all horizons exist inside `split`.

    `seasonal_max` reserves room for the same-time-last-week lookback, so the sampler
    never produces a window that would need data from before the series starts.
    """
    lo, hi = bundle.meta["split_index"][split]
    horizons = bundle.meta["horizons"]
    first = max(lo, seasonal_max, window - 1)
    last = min(hi, bundle.n_steps - max(horizons)) - 1
    if last < first:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(first, last + 1, dtype=np.int64)
    # skip anchors whose target bins are DST-distorted - Phase 1/2 measured those as
    # artefacts, and training a model to reproduce them would be learning the bug
    bad = bundle.dst_mask
    keep = np.ones(starts.shape[0], dtype=bool)
    for h in horizons:
        keep &= ~bad[starts + h]
    keep &= ~bad[starts]
    return starts[keep]


# ----------------------------------------------------------------- building blocks
class GatedResidualNetwork(nn.Module):
    """The GRN from the TFT paper: gated skip connection with optional context."""

    def __init__(self, input_size: int, hidden: int, output_size: int,
                 dropout: float = 0.1, context_size: int | None = None) -> None:
        super().__init__()
        self.project = (nn.Linear(input_size, output_size)
                        if input_size != output_size else nn.Identity())
        self.fc1 = nn.Linear(input_size, hidden)
        self.context = nn.Linear(context_size, hidden, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden, output_size)
        self.gate = nn.Linear(hidden, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc1(x)
        if self.context is not None and context is not None:
            hidden = hidden + self.context(context)
        hidden = F.elu(hidden)
        hidden = self.dropout(hidden)
        gated = torch.sigmoid(self.gate(hidden)) * self.fc2(hidden)
        return self.norm(self.project(x) + gated)


class VariableSelection(nn.Module):
    """Per-variable GRNs plus a softmax over variables - the TFT's feature gate."""

    def __init__(self, n_vars: int, hidden: int, dropout: float = 0.1,
                 context_size: int | None = None) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.per_var = nn.ModuleList(
            [GatedResidualNetwork(1, hidden, hidden, dropout) for _ in range(n_vars)]
        )
        self.weights = GatedResidualNetwork(n_vars, hidden, n_vars, dropout, context_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        # x: [..., n_vars]
        transformed = torch.stack(
            [self.per_var[i](x[..., i: i + 1]) for i in range(self.n_vars)], dim=-2
        )  # [..., n_vars, hidden]
        weights = torch.softmax(self.weights(x, context), dim=-1).unsqueeze(-1)
        return (transformed * weights).sum(dim=-2), weights.squeeze(-1)


class GraphAttentionLayer(nn.Module):
    """GAT over an explicit edge list - no torch-geometric dependency.

    Attention is normalised per destination node with a segment softmax built from
    `index_add`, which keeps the whole layer to a handful of dense ops.
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # x: [M, N, F]; edges: [2, E] (src -> dst)
        m, n, _ = x.shape
        h = self.linear(x).view(m, n, self.heads, self.out_features)
        src, dst = edges[0], edges[1]
        h_src = h[:, src]                                   # [M, E, H, F']
        h_dst = h[:, dst]
        logits = F.leaky_relu(
            (h_src * self.attn_src).sum(-1) + (h_dst * self.attn_dst).sum(-1), 0.2
        )                                                    # [M, E, H]
        # segment softmax over incoming edges of each destination node
        logits = logits - logits.max()
        weights = logits.exp()
        denom = torch.zeros(m, n, self.heads, device=x.device, dtype=weights.dtype)
        denom.index_add_(1, dst, weights)
        alpha = weights / (denom[:, dst] + 1e-16)
        alpha = self.dropout(alpha)
        out = torch.zeros(m, n, self.heads, self.out_features,
                          device=x.device, dtype=h.dtype)
        out.index_add_(1, dst, h_src * alpha.unsqueeze(-1))
        return out.reshape(m, n, self.heads * self.out_features)


# ---------------------------------------------------------------------- ST-GNN
class STGNN(nn.Module):
    """~2 GAT layers over spatial adjacency, a GRU over time, an MLP multi-horizon head.

    Output is Softplus so predictions are non-negative by construction rather than by
    clipping afterwards.
    """

    def __init__(self, n_dynamic: int, n_static: int, n_horizons: int,
                 hidden: int = 32, gru_hidden: int = 64, heads: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_dynamic, hidden)
        self.gat1 = GraphAttentionLayer(hidden, hidden // heads, heads, dropout)
        self.gat2 = GraphAttentionLayer(hidden, hidden // heads, heads, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.static_proj = nn.Linear(n_static, hidden)
        self.gru = nn.GRU(hidden, gru_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden + hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, n_horizons * 2),
        )
        self.n_horizons = n_horizons

    def forward(self, dynamic: torch.Tensor, static: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        # dynamic: [B, T, N, Fd]; static: [N, Fs]
        b, t, n, _ = dynamic.shape
        x = self.input_proj(dynamic).reshape(b * t, n, -1)
        x = self.norm1(x + F.elu(self.gat1(x, edges)))
        x = self.norm2(x + F.elu(self.gat2(x, edges)))
        x = x.reshape(b, t, n, -1).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, hidden = self.gru(x)
        hidden = hidden[-1].reshape(b, n, -1)
        static_emb = self.static_proj(static).unsqueeze(0).expand(b, -1, -1)
        out = self.head(torch.cat([hidden, static_emb], dim=-1))
        return F.softplus(out).reshape(b, n, self.n_horizons, 2)


# ------------------------------------------------------------------------- TFT
class CompactTFT(nn.Module):
    """Temporal Fusion Transformer, reduced to its load-bearing parts.

    Static encoder -> variable selection on observed inputs -> LSTM encoder ->
    static enrichment -> interpretable multi-head attention -> position-wise GRN ->
    multi-horizon head with Softplus.
    """

    def __init__(self, n_observed: int, n_known: int, n_static: int, n_horizons: int,
                 hidden: int = 64, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.static_encoder = GatedResidualNetwork(n_static, hidden, hidden, dropout)
        self.observed_selection = VariableSelection(n_observed, hidden, dropout, hidden)
        self.known_selection = VariableSelection(n_known, hidden, dropout, hidden)
        self.encoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.enrichment = GatedResidualNetwork(hidden, hidden, hidden, dropout, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                               batch_first=True)
        self.post_attention = GatedResidualNetwork(hidden, hidden, hidden, dropout)
        self.head = nn.Linear(hidden, n_horizons * 2)
        self.n_horizons = n_horizons

    def forward(self, observed: torch.Tensor, known: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        # observed/known: [B, T, F]; static: [B, Fs]
        context = self.static_encoder(static)
        obs, _ = self.observed_selection(observed, context.unsqueeze(1))
        kno, _ = self.known_selection(known, context.unsqueeze(1))
        encoded, _ = self.encoder(obs + kno)
        enriched = self.enrichment(encoded, context.unsqueeze(1))
        mask = torch.triu(
            torch.ones(enriched.size(1), enriched.size(1), device=enriched.device,
                       dtype=torch.bool), diagonal=1
        )
        attended, _ = self.attention(enriched, enriched, enriched, attn_mask=mask)
        out = self.post_attention(attended + enriched)[:, -1]
        return F.softplus(self.head(out)).reshape(-1, self.n_horizons, 2)


# ------------------------------------------------------------------------ loss
def poisson_nll(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson negative log-likelihood, the natural loss for bounded count data."""
    rate = prediction.clamp_min(1e-6)
    return (rate - target * torch.log(rate)).mean()
