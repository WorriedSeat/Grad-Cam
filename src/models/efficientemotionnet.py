import torch
import torch.nn as nn
import torchvision.models as models


class EfficientEmotionNet(nn.Module):
    """
    EfficientNet-B0 pretrained on ImageNet, fine-tuned for 7-class emotion recognition.
    GradCAM target layer: 'model.features.8'
    """
    def __init__(self, num_classes: int = 7, dropout: float = 0.4):
        super().__init__()
        backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        for name, param in backbone.named_parameters():
            if "features.0." in name or "features.1." in name:
                param.requires_grad = False
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)