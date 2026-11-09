"""Frozen ImageNet DenseNet-121 used as a fixed feature extractor (no training, no fine-tuning).

The ImageNet weights are read from the local torch cache (~/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth);
torchvision only downloads them if that file is missing."""
import numpy as np
import torch
import torchvision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_densenet(device):
    weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    model = torchvision.models.densenet121(weights=weights)
    model.classifier = torch.nn.Identity()          # output = the 1,024-d pooled feature vector
    return model.eval().to(device)


@torch.no_grad()
def dense_features(model, arrs224, device, batch=64):
    """(N,224,224) uint8 grayscale -> (N,1024) float32."""
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(arrs224), batch):
        x = torch.from_numpy(np.ascontiguousarray(arrs224[i:i + batch])).float().div(255).to(device)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)
