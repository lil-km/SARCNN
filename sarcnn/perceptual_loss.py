import numpy as np
import torch
from torch import nn
from torchvision.models import vgg19


class PERCEPTUAL_LOSS(nn.Module):
    """Constructs a perceptual loss function based on the VGG19 network.
     """

    def __init__(self) -> None:
        super(PERCEPTUAL_LOSS, self).__init__()
        # Load the VGG19 model trained on the ImageNet dataset.
        vgg19_model = vgg19(pretrained=True).eval()
        # Extract the thirty-sixth layer output in the VGG19 model as the perceptual loss.
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])
        # Freeze model parameters.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.criterion = nn.L1Loss()

        # The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, denoised_image: torch.Tensor, ground_truth_image: torch.Tensor) -> torch.Tensor:
        # Standardized operations
        denoised_image = denoised_image.sub(self.mean).div(self.std)
        ground_truth_image = ground_truth_image.sub(self.mean).div(self.std)

        # Find the feature map difference between the two images
        loss = self.criterion(self.feature_extractor(denoised_image), self.feature_extractor(ground_truth_image))

        return loss
