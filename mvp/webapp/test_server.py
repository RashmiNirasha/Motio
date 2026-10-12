import json
import numpy as np
import requests

SERVER = "http://127.0.0.1:8000"

X = np.load(
    "mvp/dataset_v2/X_test.npy"
).astype(np.float32)

y_original = np.load(
    "mvp/dataset_v2/y_test.npy"
)

# V2 labels:
# 0 = background
# 1 = object_interaction
# 2 = point

interaction_indices = np.where(
    y_original == 1
)[0]

no_interaction_indices = np.where(
    y_original != 1
)[0]

interaction_index = int(
    interaction_indices[0]
)

no_interaction_index = int(
    no_interaction_indices[0]
)


def test_sample(name, index):

    sample = X[index]

    payload = {
        "sequence": sample.tolist()
    }

    response = requests.post(
        SERVER + "/predict",
        json=payload,
        timeout=10
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Dataset index:", index)
    print("Original V2 label:", y_original[index])
    print("HTTP status:", response.status_code)

    print(
        json.dumps(
            response.json(),
            indent=2
        )
    )


test_sample(
    "REAL OBJECT INTERACTION",
    interaction_index
)

test_sample(
    "REAL NO-INTERACTION",
    no_interaction_index
)
