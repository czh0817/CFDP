import torch
import torch.nn as nn


class CrossDomainFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Tanh()
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, src_feat, tgt_feat):
        src_feat = src_feat.unsqueeze(1)
        tgt_feat = tgt_feat.unsqueeze(1)
        src_attn, _ = self.cross_att(src_feat, tgt_feat, tgt_feat)
        tgt_attn, _ = self.cross_att(tgt_feat, src_feat, src_feat)
        src_feat = src_feat + src_attn
        tgt_feat = tgt_feat + tgt_attn
        src_gate = self.gate(src_feat.squeeze(1)) + 1
        tgt_gate = self.gate(tgt_feat.squeeze(1)) + 1
        src_out = self.norm(src_feat.squeeze(1) + src_gate * src_attn.squeeze(1))
        tgt_out = self.norm(tgt_feat.squeeze(1) + tgt_gate * tgt_attn.squeeze(1))
        src_out = src_out + self.feed_forward(src_out)
        tgt_out = tgt_out + self.feed_forward(tgt_out)

        return src_out, tgt_out