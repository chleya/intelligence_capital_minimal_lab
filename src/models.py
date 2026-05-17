import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StateOnlyPredictor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.bottleneck = MLP(self.in_dim, [bottleneck_dim * 2, bottleneck_dim], bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, n_actions)

    def forward(self, x):
        z = self.bottleneck(x)
        return self.head(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class ActionOnlyPredictor(nn.Module):
    def __init__(self, n_actions=3):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_actions))

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.shape[0], -1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class AEPCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
        self.action_encoder = nn.Embedding(n_actions, bottleneck_dim // 2)
        self.decoder = MLP(bottleneck_dim + bottleneck_dim // 2, [bottleneck_dim], bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, 1)

    def forward(self, x, action_idx):
        z_state = self.state_encoder(x)
        z_action = self.action_encoder(action_idx)
        z = torch.cat([z_state, z_action], dim=-1)
        z = self.decoder(z)
        return self.head(z)

    def predict_all_actions(self, x):
        z_state = self.state_encoder(x)
        outputs = []
        for a in range(3):
            a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
            z_action = self.action_encoder(a_t)
            z = torch.cat([z_state, z_action], dim=-1)
            z = self.decoder(z)
            outputs.append(self.head(z))
        return torch.cat(outputs, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class ResidualCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16, residual_dim=8):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
        self.autonomous_head = nn.Linear(bottleneck_dim, 1)
        self.action_encoder = nn.Embedding(n_actions, residual_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(bottleneck_dim + residual_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )

    def forward(self, x, action_idx):
        z_state = self.state_encoder(x)
        b = self.autonomous_head(z_state)
        z_action = self.action_encoder(action_idx)
        r = self.residual_head(torch.cat([z_state, z_action], dim=-1))
        return b + r, b, r

    def predict_all_actions(self, x):
        z_state = self.state_encoder(x)
        b = self.autonomous_head(z_state)
        outputs = []
        for a in range(3):
            a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
            z_action = self.action_encoder(a_t)
            r = self.residual_head(torch.cat([z_state, z_action], dim=-1))
            outputs.append(b + r)
        return torch.cat(outputs, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CenteredResidualCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16, residual_dim=8):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
        self.mean_head = nn.Linear(bottleneck_dim, 1)
        self.action_encoder = nn.Embedding(n_actions, residual_dim)
        self.centered_residual_head = nn.Sequential(
            nn.Linear(bottleneck_dim + residual_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )

    def forward(self, x, action_idx):
        z_state = self.state_encoder(x)
        m = self.mean_head(z_state)
        z_action = self.action_encoder(action_idx)
        cr = self.centered_residual_head(torch.cat([z_state, z_action], dim=-1))
        return m + cr, m, cr

    def predict_all_actions(self, x):
        z_state = self.state_encoder(x)
        m = self.mean_head(z_state)
        outputs = []
        for a in range(3):
            a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
            z_action = self.action_encoder(a_t)
            cr = self.centered_residual_head(torch.cat([z_state, z_action], dim=-1))
            outputs.append(m + cr)
        return torch.cat(outputs, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CounterfactualCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.encoder = MLP(self.in_dim, [bottleneck_dim * 2, bottleneck_dim], bottleneck_dim)
        self.head = nn.Linear(bottleneck_dim, n_actions)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class CausalContrastCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16, temperature=0.1):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
        self.head = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim // 2), nn.ReLU(), nn.Linear(bottleneck_dim // 2, n_actions))
        self.temperature = temperature

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z), z

    def contrastive_loss(self, z, labels):
        z_norm = F.normalize(z, dim=1)
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos = (sim * labels_eq).sum(1) / (labels_eq.sum(1) + 1e-8)
        neg = (sim * (1 - labels_eq)).sum(1) / ((1 - labels_eq).sum(1) + 1e-8)
        return (-pos + torch.logsumexp(sim, dim=1)).mean()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


class ResidualAdversarialCompressor(nn.Module):
    def __init__(self, obs_dim, history_len, n_actions=3, bottleneck_dim=16, residual_dim=8):
        super().__init__()
        self.in_dim = history_len * (obs_dim + 1)
        self.state_encoder = MLP(self.in_dim, [bottleneck_dim * 2], bottleneck_dim)
        self.autonomous_head = nn.Linear(bottleneck_dim, 1)
        self.action_encoder = nn.Embedding(n_actions, residual_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(bottleneck_dim + residual_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )
        self.adversary = nn.Sequential(
            nn.Linear(bottleneck_dim + residual_dim, bottleneck_dim // 2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )

    def forward(self, x, action_idx):
        z_state = self.state_encoder(x)
        b = self.autonomous_head(z_state)
        z_action = self.action_encoder(action_idx)
        r_features = torch.cat([z_state, z_action], dim=-1)
        r = self.residual_head(r_features)
        adv = self.adversary(r_features.detach())
        return b + r, b, r, adv

    def predict_all_actions(self, x):
        z_state = self.state_encoder(x)
        b = self.autonomous_head(z_state)
        outputs = []
        for a in range(3):
            a_t = torch.full((x.shape[0],), a, dtype=torch.long, device=x.device)
            z_action = self.action_encoder(a_t)
            r_features = torch.cat([z_state, z_action], dim=-1)
            r = self.residual_head(r_features)
            outputs.append(b + r)
        return torch.cat(outputs, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())