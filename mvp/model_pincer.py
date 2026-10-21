import torch
import torch.nn as nn


class PincerTemporalNet(nn.Module):

    def __init__(
        self,
        feature_mean,
        feature_std
    ):
        super().__init__()

        feature_dim = len(feature_mean)

        self.register_buffer(
            "feature_mean",
            torch.as_tensor(
                feature_mean,
                dtype=torch.float32
            ).reshape(1, 1, feature_dim)
        )

        self.register_buffer(
            "feature_std",
            torch.as_tensor(
                feature_std,
                dtype=torch.float32
            ).reshape(1, 1, feature_dim)
        )

        # Normalized features + frame-to-frame motion,
        # same idea as BinaryTemporalHandNet but smaller,
        # because the recorded pincer dataset is small.

        self.temporal = nn.Sequential(

            nn.Conv1d(
                feature_dim * 2,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Conv1d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        # x:
        # B × T × feature_dim

        x = (
            x - self.feature_mean
        ) / self.feature_std

        delta = torch.zeros_like(x)

        delta[:, 1:, :] = (
            x[:, 1:, :]
            -
            x[:, :-1, :]
        )

        x = torch.cat(
            [x, delta],
            dim=2
        ).transpose(1, 2)

        x = self.temporal(x)

        return self.classifier(x).squeeze(1)
