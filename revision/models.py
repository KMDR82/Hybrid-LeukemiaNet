"""Models compared in the revision.

hybrid   EfficientNet-B0 + ViT-B/16, features concatenated (1280 + 768 = 2048)
cnn      EfficientNet-B0 alone (1280)       -- trained independently
vit      ViT-B/16 alone (768)               -- trained independently

All three share the same classifier head design (D -> 512 -> 128 -> 2) so the
comparison isolates the backbone(s). Backbones start from ImageNet-1k weights
(torchvision ``DEFAULT``) and NO layer is frozen: every parameter is trained.

The hybrid model keeps the stream-masking switches of the original submission.
In the revision they are used only for the secondary "stream masking" analysis
(zeroing one feature stream of the trained hybrid at inference); they are not a
substitute for the independently trained single-backbone models.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

CNN_DIM, VIT_DIM = 1280, 768


def classifier_head(in_dim, num_classes=2, dropout=0.4):
    return nn.Sequential(
        nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def _efficientnet(pretrained):
    m = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
    m.classifier = nn.Identity()
    return m


def _vit(pretrained):
    m = tvm.vit_b_16(weights="DEFAULT" if pretrained else None)
    m.heads = nn.Identity()
    return m


class LeukemiaNet(nn.Module):
    def __init__(self, arch="hybrid", num_classes=2, dropout=0.4, pretrained=True):
        super().__init__()
        if arch not in ("hybrid", "cnn", "vit"):
            raise ValueError(arch)
        self.arch = arch
        self.cnn = _efficientnet(pretrained) if arch in ("hybrid", "cnn") else None
        self.vit = _vit(pretrained) if arch in ("hybrid", "vit") else None
        in_dim = (CNN_DIM if self.cnn is not None else 0) + (VIT_DIM if self.vit is not None else 0)
        self.classifier = classifier_head(in_dim, num_classes, dropout)
        # stream-masking switches (hybrid only, inference-time analysis)
        self.mask_cnn = False
        self.mask_vit = False

    def features(self, x):
        parts = []
        if self.cnn is not None:
            f = self.cnn(x)
            parts.append(torch.zeros_like(f) if self.mask_cnn else f)
        if self.vit is not None:
            f = self.vit(x)
            parts.append(torch.zeros_like(f) if self.mask_vit else f)
        return torch.cat(parts, dim=1)

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
