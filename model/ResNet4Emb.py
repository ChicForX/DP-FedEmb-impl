import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps)


class ResNet18Client(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=10, global_avg_pool=True):
        super(ResNet18Client, self).__init__()
        # load pretrained resnet18
        base_model = models.resnet18()
        layers = list(base_model.children())[:-2]  # remove fc & pooling layers

        # replace BatchNorm with GroupNorm
        for i in range(len(layers)):
            if isinstance(layers[i], nn.BatchNorm2d):
                layers[i] = GroupNorm(layers[i].num_features)

        self.features = nn.Sequential(*layers)
        self.global_avg_pool = global_avg_pool
        self.embedding = nn.Linear(512, embedding_dim)  # Adjusted for ResNet-18

        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        if self.global_avg_pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        embedding = self.embedding(x)

        classes = self.classifier(embedding)
        return classes, embedding


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())[:-2]  # remove fc & pooling layers

        # replace BatchNorm with GroupNorm
        for i in range(len(layers)):
            if isinstance(layers[i], nn.BatchNorm2d):
                layers[i] = GroupNorm(layers[i].num_features)

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


class ResNet18BackboneWithClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18BackboneWithClassifier, self).__init__()
        # Load the backbone
        self.backbone = ResNet18Backbone()

        # Add a classifier on top of the backbone
        self.classifier = nn.Linear(512, num_classes)  # Adjust the dimensions as needed

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)
