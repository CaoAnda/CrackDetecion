from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Net(nn.Module):
    def __init__(self,backbone_model='resnet18', backbone_pretrained=True, **kwargs) -> None:
        super().__init__()
        self.backbone = getattr(models, backbone_model)(
            weights='IMAGENET1K_V1' if backbone_pretrained else None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, A, B, *args):
        _A = self.backbone(A)
        _B = self.backbone(B)
        
        x = torch.abs(_A - _B).flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = torch.sigmoid(x)
        return x

if __name__ == '__main__':
    model = Net()
    A = torch.rand((2, 3, 64, 64))
    B = torch.rand((2, 3, 64, 64))
    out = model(A, B)
    print(out)
        