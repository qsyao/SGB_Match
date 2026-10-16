from .segment_anything import build_sam_vit_b
import torch
import torch.nn as nn
import torch.nn.functional as F

# test_model = build_sam_vit_b(image_size=384).image_encoder

# test_input = torch.rand(1, 3, 384, 384)

# test_output = test_model(test_input)

class Base_SAM_ViT(nn.Module):
    def __init__(self, image_size, use_mlp=True):
        super().__init__()
        self.image_encoder = build_sam_vit_b(image_size).image_encoder

        self.MLP = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, 1, 1),
        )

        pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1).cuda()
        pixel_std = [58.395, 57.12, 57.375]
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1).cuda()
    
    def forward_feature(self, x):
        x = (x * 255 - self.pixel_mean) / self.pixel_std
        with torch.no_grad():
            x = self.image_encoder(x)
        return x


