import torch
import torch.nn as nn
import torchvision.models as models

class DynamicLeukemiaNet(nn.Module):
    """
    Dual-Stream architecture combining EfficientNet-B0 and ViT-B_16.
    Includes dynamic gating for ablation studies.
    """
    def __init__(self, num_classes=2, dropout_rate=0.4):
        super(DynamicLeukemiaNet, self).__init__()
        
        # CNN Stream (EfficientNet-B0)
        # weights='DEFAULT' ile en güncel ImageNet ağırlıkları çekilir
        self.cnn = models.efficientnet_b0(weights='DEFAULT')
        cnn_out_features = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()
        
        # ViT Stream (ViT-B_16)
        self.vit = models.vit_b_16(weights='DEFAULT')
        vit_out_features = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        # Fusion and Classification Block
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_features + vit_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes) 
        )
        
        # Ablation switches
        self.use_cnn = True
        self.use_vit = True

    def forward(self, x):
        # Feature extraction with ablation logic
        cnn_feat = self.cnn(x) if self.use_cnn else torch.zeros(x.size(0), 1280).to(x.device)
        vit_feat = self.vit(x) if self.use_vit else torch.zeros(x.size(0), 768).to(x.device)
        
        # Feature fusion
        fused_features = torch.cat((cnn_feat, vit_feat), dim=1)
        return self.classifier(fused_features)