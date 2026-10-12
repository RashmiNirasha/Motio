import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

sys.path.append("mvp")

from model_binary import BinaryTemporalHandNet


CHECKPOINT_PATH = (
    "mvp/models_binary/"
    "child_object_interaction.pt"
)

OUTPUT_PATH = (
    "mvp/coreml/"
    "ChildObjectInteraction.mlpackage"
)

TEST_DATA = (
    "mvp/dataset_v2/"
    "X_test.npy"
)

THRESHOLD_PATH = (
    "mvp/models_binary/"
    "iphone_threshold.txt"
)


print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)


feature_mean = checkpoint[
    "feature_mean"
]

feature_std = checkpoint[
    "feature_std"
]


base_model = BinaryTemporalHandNet(
    feature_mean,
    feature_std
)

base_model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

base_model.eval()


# --------------------------------------------------
# Wrapper:
# Return probability rather than raw logit.
# This makes Swift integration simpler.
# --------------------------------------------------

class ProbabilityModel(nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, x):

        logits = self.model(x)

        probability = torch.sigmoid(
            logits
        )

        # Core ML output:
        # batch × 1
        return probability.unsqueeze(1)


model = ProbabilityModel(
    base_model
)

model.eval()


# --------------------------------------------------
# Trace
# --------------------------------------------------

example_input = torch.zeros(
    1,
    20,
    69,
    dtype=torch.float32
)


print("Tracing PyTorch model...")

traced_model = torch.jit.trace(
    model,
    example_input
)


# Quick TorchScript validation
with torch.no_grad():

    original = model(
        example_input
    )

    traced = traced_model(
        example_input
    )


print(
    "PyTorch test output:",
    original.numpy()
)

print(
    "TorchScript test output:",
    traced.numpy()
)


# --------------------------------------------------
# Convert to Core ML
# --------------------------------------------------

print("\nConverting to Core ML...")


mlmodel = ct.convert(

    traced_model,

    convert_to="mlprogram",

    inputs=[
        ct.TensorType(
            name="hand_sequence",
            shape=(1, 20, 69),
            dtype=np.float32
        )
    ],

    outputs=[
        ct.TensorType(
            name="interaction_probability",
            dtype=np.float32
        )
    ],

    minimum_deployment_target=(
        ct.target.iOS17
    ),

    compute_precision=(
        ct.precision.FLOAT32
    )
)


mlmodel.short_description = (
    "Detects child hand-object interaction "
    "from a 20-frame hand landmark sequence."
)


mlmodel.input_description[
    "hand_sequence"
] = (
    "Shape [1,20,69]. "
    "Twenty temporal frames of normalized "
    "hand landmark and motion-related features."
)


mlmodel.output_description[
    "interaction_probability"
] = (
    "Probability of object interaction."
)


# --------------------------------------------------
# Save
# --------------------------------------------------

mlmodel.save(
    OUTPUT_PATH
)


print(
    "\nSaved:",
    OUTPUT_PATH
)


# --------------------------------------------------
# Save app configuration
# --------------------------------------------------

with open(
    THRESHOLD_PATH,
    "r"
) as f:

    threshold = float(
        f.read().strip()
    )


config = {

    "sequence_length": 20,

    "feature_dimension": 69,

    "interaction_threshold": threshold,

    # Later, phone must see multiple
    # positives before displaying alert.
    "confirmation_windows": 3
}


with open(
    "mvp/coreml/model_config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=2
    )


print(
    "Saved: "
    "mvp/coreml/model_config.json"
)


# --------------------------------------------------
# Compare PyTorch vs Core ML
# --------------------------------------------------

print(
    "\nTesting converted model..."
)


X_test = np.load(
    TEST_DATA
).astype(np.float32)


sample = X_test[
    0:1
]


with torch.no_grad():

    torch_probability = (
        model(
            torch.from_numpy(sample)
        )
        .numpy()
        .reshape(-1)[0]
    )


print(
    "PyTorch probability:",
    float(torch_probability)
)


try:

    prediction = mlmodel.predict(
        {
            "hand_sequence":
                sample
        }
    )

    coreml_output = prediction[
        "interaction_probability"
    ]

    coreml_probability = float(
        np.asarray(
            coreml_output
        ).reshape(-1)[0]
    )

    print(
        "Core ML probability:",
        coreml_probability
    )

    print(
        "Difference:",
        abs(
            float(torch_probability)
            -
            coreml_probability
        )
    )

except Exception as e:

    print(
        "\nCore ML model was converted "
        "successfully, but local prediction "
        "could not be executed:"
    )

    print(e)


print("\nDONE.")
