import torch
from torch import nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    
    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # 70x70x3 -> 16x16x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #8x8x64

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2), # 8x8x192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), # 4x4x192

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1), #4x4x384
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1), #4x4x256
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), #4x4x256
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=3, stride=2), # 2x2x256
            # Commented the above line about because it caused the dimensions to shrink too much and threw an error. 
            # Figured the Adaptive Average Pool would do the Same Job.

            nn.AdaptiveAvgPool2d((6, 6)), # 6x6x256

            nn.Flatten(),

            nn.Dropout(p=0.5),
            nn.Linear(9216, 4096),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x

