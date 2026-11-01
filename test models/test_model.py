import os
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl",
    "sign_lstm_torch.pth"
)

LABEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isl",
    "labels.npy"
)


# ============================================================
# CHECK FILES
# ============================================================

print("Model path:")
print(MODEL_PATH)

print()

print("Label path:")
print(LABEL_PATH)

print()


if not os.path.exists(MODEL_PATH):
    print("ERROR: sign_lstm_torch.pth not found")
    exit()

if not os.path.exists(LABEL_PATH):
    print("ERROR: labels.npy not found")
    exit()


print("Both model files found.")


# ============================================================
# LOAD LABELS
# ============================================================

labels = np.load(
    LABEL_PATH,
    allow_pickle=True
)

print()
print("Labels:")
print(labels)

print()
print("Number of classes:", len(labels))


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=42,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(
            64,
            num_classes
        )


    def forward(self, x):

        _, (hn, _) = self.lstm(x)

        return self.fc(hn[-1])


# ============================================================
# LOAD MODEL
# ============================================================

model = LSTMModel(len(labels))

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


print()
print("================================")
print("ISL MODEL LOADED SUCCESSFULLY")
print("================================")