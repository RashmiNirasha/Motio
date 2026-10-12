import torch
import torch.nn as nn


class TemporalHandNet(nn.Module):

    def __init__(
        self,
        feature_mean,
        feature_std,
        num_classes=6
    ):
        super().__init__()

        mean = torch.as_tensor(
            feature_mean,
            dtype=torch.float32
        ).reshape(1, 1, 63)

        std = torch.as_tensor(
            feature_std,
            dtype=torch.float32
        ).reshape(1, 1, 63)

        self.register_buffer(
            "feature_mean",
            mean
        )

        self.register_buffer(
            "feature_std",
            std
        )

        self.temporal = nn.Sequential(

            nn.Conv1d(
                63,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                96,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                96,
                96,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                96,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # x:
        # batch × 20 frames × 63 features

        x = (
            x - self.feature_mean
        ) / self.feature_std

        # Conv1D expects:
        # batch × features × time
        x = x.transpose(1, 2)

        x = self.temporal(x)

        return self.classifier(x)
