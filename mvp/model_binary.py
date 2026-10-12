import torch
import torch.nn as nn


class BinaryTemporalHandNet(nn.Module):

    def __init__(
        self,
        feature_mean,
        feature_std
    ):
        super().__init__()

        mean = torch.as_tensor(
            feature_mean,
            dtype=torch.float32
        ).reshape(1, 1, 69)

        std = torch.as_tensor(
            feature_std,
            dtype=torch.float32
        ).reshape(1, 1, 69)

        self.register_buffer(
            "feature_mean",
            mean
        )

        self.register_buffer(
            "feature_std",
            std
        )

        # Input becomes:
        #
        # 69 original normalized features
        # +
        # 69 frame-to-frame motion features
        #
        # = 138 channels

        self.temporal = nn.Sequential(

            nn.Conv1d(
                138,
                96,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                96,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        # x:
        # B × 20 × 69

        x = (
            x - self.feature_mean
        ) / self.feature_std

        # Explicit motion/delta features
        delta = torch.zeros_like(x)

        delta[:, 1:, :] = (
            x[:, 1:, :]
            -
            x[:, :-1, :]
        )

        x = torch.cat(
            [x, delta],
            dim=2
        )

        # B × 138 × 20
        x = x.transpose(1, 2)

        x = self.temporal(x)

        logits = self.classifier(x)

        return logits.squeeze(1)
