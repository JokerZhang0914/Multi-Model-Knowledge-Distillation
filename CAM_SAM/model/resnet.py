import torch
import torch.nn as nn



class PretrainedResNet18_Encoder(nn.Module):
    def __init__(self, freeze=False):
        super(PretrainedResNet18_Encoder, self).__init__()
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

class Student_Head(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super(Student_Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class FullResNet(nn.Module):
    def __init__(self, encoder, student):
        super().__init__()
        self.encoder = encoder
        self.student = student

    def forward(self, x):
        feats = self.encoder(x)
        return self.student(feats)