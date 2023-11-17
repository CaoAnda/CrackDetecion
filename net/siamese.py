import ssl
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models
ssl._create_default_https_context = ssl._create_unverified_context

def get_feature_channel(model):
    inputs = torch.rand(1, 3, 64, 64)
    outputs = model(inputs).flatten(1)
    return outputs.shape[1]

class Net(nn.Module):
    def __init__(self,backbone_model='resnet18', backbone_pretrained=True, **kwargs) -> None:
        super().__init__()
        self.backbone = getattr(models, backbone_model)(
            weights='IMAGENET1K_V1' if backbone_pretrained else None)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        feature_length = get_feature_channel(self.backbone)
        self.fc1 = nn.Linear(feature_length, feature_length // 2)
        self.fc2 = nn.Linear(feature_length // 2, 1)
        
    def forward(self, A, B, *args):
        _A = self.get_embedding(A)
        _B = self.get_embedding(B)
        
        # x = torch.abs(_A - _B).flatten(1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        
        # x = torch.sigmoid(x)
        x = torch.cosine_similarity(_A, _B, dim=1).unsqueeze(1)
        return x
    
    def get_embedding(self, x):
        return self.backbone(x).flatten(1)

if __name__ == '__main__':
    model = Net(backbone_model='resnet18')
    A = torch.rand((2, 3, 64, 64))
    B = torch.rand((2, 3, 64, 64))
    out = model(A, B)
    print(out)
    print(f"==>> out.shape: {out.shape}")
        