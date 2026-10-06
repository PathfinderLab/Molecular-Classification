import numpy as np
import torch
from torch import nn
from .transMIL import TransLayer, PPEG
from .transformer import MutiHeadAttention, MutiHeadAttention_modify
from .network import Classifier_1fc


class TransMIL_MBA(nn.Module):
    """TransMIL backbone with Multi-Branch Attention aggregator."""

    def __init__(self, conf, n_token=1, n_masked_patch=0, mask_drop=0):
        super().__init__()
        self.n_token = n_token
        self.pos_layer = PPEG(dim=conf.D_inner)
        self._fc1 = nn.Sequential(nn.Linear(conf.D_feat, conf.D_inner), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, conf.D_inner))
        self.layer1 = TransLayer(dim=conf.D_inner)
        self.layer2 = TransLayer(dim=conf.D_inner)
        self.norm = nn.LayerNorm(conf.D_inner)

        self.sub_attention = nn.ModuleList(
            [MutiHeadAttention(conf.D_inner, 8, n_masked_patch=n_masked_patch,
                                mask_drop=mask_drop) for _ in range(n_token)]
        )
        self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)

        self.classifier = nn.ModuleList(
            [Classifier_1fc(conf.D_inner, conf.n_class, 0.0) for _ in range(n_token)]
        )
        self.slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input):
        h = self._fc1(input)
        H = h.shape[1]
        _H = int(np.ceil(np.sqrt(H)))
        _W = int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)
        h = self.norm(h)

        patch_feat = h[:, 1:]
        q = self.q
        k = patch_feat
        v = patch_feat

        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)
        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))
        slide_out = self.slide_classifier(feat_bag)
        return torch.cat(outputs, dim=0), slide_out, attns

