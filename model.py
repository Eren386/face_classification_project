import torch.nn as nn
from torchvision import models


class FaceClassifier(nn.Module):
    """Binary face classifier built on top of a pretrained ResNet-18 backbone."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    print("Testing FaceClassifier initialization...")
    classifier = FaceClassifier(num_classes=2)
    print("Model initialized successfully!")
    print("Final classification layer:", classifier.model.fc)
