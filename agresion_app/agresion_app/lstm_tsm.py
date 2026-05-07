"""
models/lstm_tsm.py  (con BiLSTMClassifierV1 para compatibilidad de checkpoints)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PERSONS    = 2
N_KP         = 17
KP_DIM       = 5
INPUT_SIZE   = N_PERSONS * N_KP * KP_DIM   # 170
DIST_ENC_DIM = 32


def _interperson_features(x: torch.Tensor) -> torch.Tensor:
    p0_c = x[:, :, 0, :, :2].mean(dim=2)
    p1_c = x[:, :, 1, :, :2].mean(dim=2)
    dist  = (p0_c - p1_c).norm(dim=-1)
    dist_v = torch.diff(dist, dim=1, prepend=dist[:, :1])
    return torch.stack([dist.mean(1), dist.min(1).values, dist_v.abs().mean(1)], dim=1)


class TemporalAttentionPool(nn.Module):
    def __init__(self, feat_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


# ── V1: sin dist_encoder — compatible con checkpoints antiguos ──────────────
class BiLSTMClassifierV1(nn.Module):
    """classifier input = hidden*2 = 512  (sin dist_encoder)"""

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 256,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, V, C = x.shape
        x_flat = x.reshape(B, T, P * V * C)
        x_norm = self.input_bn(x_flat.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_norm)
        att_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * att_w).sum(dim=1)
        return self.classifier(context)


# ── V2: con dist_encoder — classifier input = 544 ───────────────────────────
class BiLSTMClassifier(nn.Module):
    """classifier input = hidden*2 + DIST_ENC_DIM = 544"""

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 256,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dist_encoder = nn.Sequential(
            nn.Linear(3, DIST_ENC_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2 + DIST_ENC_DIM, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, V, C = x.shape
        dist_f   = _interperson_features(x)
        dist_enc = self.dist_encoder(dist_f)
        x_flat   = x.reshape(B, T, P * V * C)
        x_norm   = self.input_bn(x_flat.permute(0, 2, 1)).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_norm)
        att_w    = torch.softmax(self.attention(lstm_out), dim=1)
        context  = (lstm_out * att_w).sum(dim=1)
        feat     = torch.cat([context, dist_enc], dim=1)
        return self.classifier(feat)


# ── TSM ──────────────────────────────────────────────────────────────────────
class TemporalShift(nn.Module):
    def __init__(self, shift_div: int = 8):
        super().__init__()
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        fold = C // self.shift_div
        out  = x.clone()
        out[:, 1:,  :fold]       = x[:, :-1, :fold]
        out[:, 0,   :fold]       = 0.0
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        out[:, -1,  fold:2*fold] = 0.0
        return out


class TSMClassifier(nn.Module):
    def __init__(self, input_size: int = INPUT_SIZE, num_classes: int = 2,
                 hidden_dim: int = 256, dropout: float = 0.4, shift_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(shift_div=shift_div)
        self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout))
        self.layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout))
        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                    nn.LayerNorm(hidden_dim * 2), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout))
        self.temp_pool   = TemporalAttentionPool(feat_dim=hidden_dim * 2)
        self.dist_encoder = nn.Sequential(nn.Linear(3, DIST_ENC_DIM),
                                          nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5))
        self.classifier  = nn.Sequential(nn.Linear(hidden_dim * 2 + DIST_ENC_DIM, hidden_dim),
                                         nn.ReLU(inplace=True), nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, V, C = x.shape
        dist_f   = _interperson_features(x)
        dist_enc = self.dist_encoder(dist_f)
        x_flat   = x.reshape(B, T, P * V * C)
        x_flat   = self.shift(x_flat)
        feat     = self.input_proj(x_flat)
        feat     = self.shift(feat);  feat = self.layer1(feat)
        feat     = self.shift(feat);  feat = self.layer2(feat)
        feat     = self.shift(feat);  feat = self.layer3(feat)
        feat_p   = self.temp_pool(feat)
        return self.classifier(torch.cat([feat_p, dist_enc], dim=1))
