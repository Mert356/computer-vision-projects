from torch import nn


class PetCNN(nn.Module):
    """
    Cat/dog CNN trained from scratch that outputs one logit per image (logit >= 0 means dog)

    Pipeline for a 128x128 input (B is the batch size):
      Input                                              [B, 3, 128, 128]
      Block 1: Conv3x3 -> BatchNorm -> ReLU -> MaxPool   [B, 32, 64, 64]
      Block 2: Conv3x3 -> BatchNorm -> ReLU -> MaxPool   [B, 64, 32, 32]
      Block 3: Conv3x3 -> BatchNorm -> ReLU -> MaxPool   [B, 128, 16, 16]
      Block 4: Conv3x3 -> BatchNorm -> ReLU -> MaxPool   [B, 256, 8, 8]
      AdaptiveAvgPool 4x4                                [B, 256, 4, 4]
      Flatten -> Dropout                                 [B, 4096]
      Linear -> ReLU -> Dropout                          [B, 128]
      Linear -> squeeze                                  [B]
    """

    def __init__(self, dropout=0.3):
        super().__init__()
        layers = []
        channels = (3, 32, 64, 128, 256)
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.extend([
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
        layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.features = nn.Sequential(*layers)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.linear(self.features(x)).squeeze(-1)
