import torch
import torch.nn as nn
import torch.nn.functional as F
from grl import GRL


class ACA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, num_domains=2):
        super().__init__()
        self.num_domains = num_domains

        self.stats_encoder = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

        self.domain_predictor = nn.Linear(in_channels, num_domains)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, domain_label=None):
        if x.dim() == 4:
            channel_mean = x.mean(dim=[2, 3])
            channel_std = torch.std(x, dim=[2, 3])
        else:
            channel_mean = x
            channel_std = torch.zeros_like(x)

        stats = torch.cat([channel_mean, channel_std], dim=1)
        base_weights = self.stats_encoder(stats)

        if domain_label is not None and self.training:
            domain_logits = self.domain_predictor(GRL.apply(channel_mean.detach(), alpha=1.0))
            domain_loss = F.cross_entropy(domain_logits, domain_label)
            domain_grad = torch.autograd.grad(domain_loss, channel_mean, retain_graph=True)[0]
            grad_weights = 1 - torch.sigmoid(domain_grad.abs().mean(dim=0, keepdim=True))
            modulated_weights = base_weights * grad_weights
            self.temperature.data = torch.clamp(self.temperature, 0.5, 2.0)
            attention = F.softmax(modulated_weights / self.temperature, dim=1)
        else:
            attention = base_weights

        if x.dim() == 4:
            return x * attention.unsqueeze(-1).unsqueeze(-1)
        else:
            return x * attention