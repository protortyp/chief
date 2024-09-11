import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn


def get_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def process_folder(model: nn.Module, folder_path: str) -> torch.Tensor:
    transforms = get_transforms()
    features = []
    device = next(model.parameters()).device
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = np.array(image)
            image_tensor = (
                transforms(Image.fromarray(image.astype("uint8"), "RGB"))
                .unsqueeze(0)
                .to(device)
            )
            with torch.no_grad():
                feat = model(image_tensor).squeeze()
            features.append(feat.cpu())
    return torch.stack(features)
