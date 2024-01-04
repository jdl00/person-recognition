import os

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from dataclasses import dataclass

from model.model import MultiOutputConv

MODEL_PATH = os.path.join(
    os.path.abspath("model"), "output", "best_val_loss_model_24.pth"
)


# Ethnicity labels
ETHNICITIES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Hispanic"}

# Gender labels
GENDERS = {0: "Male", 1: "Female"}

# Age labels
AGES = {0: "0-18", 1: "18-35", 2: "35-50", 3: "50-65", 4: "65-90", 5: "90+"}


model = MultiOutputConv()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


@dataclass
class Result:
    class_name: str
    prob: float
    category: str

    def get_output(self):
        self.prob = float(self.prob)
        return f"{self.category} {self.class_name} Probability: {self.prob:.2f}"


def preprocess(frame, size=(48, 48)):
    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

    # Resize the frame
    resized_frame = cv2.resize(gray_frame, size)

    # Convert to a float tensor and scale to [0, 1]
    frame_tensor = torch.tensor(resized_frame).float() / 255.0

    # Add a channel dimension (C, H, W) and a batch dimension (N, C, H, W)
    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)

    return frame_tensor


def infer(frame):
    preprocessed_frame = preprocess(frame)

    results = []
    # Perform inference
    with torch.no_grad():
        (age_logits, ethnicity_logits, gender_logits) = model(preprocessed_frame)

        # Get class probabilities
        age_probs = F.softmax(age_logits, dim=1)
        ethnicity_probs = F.softmax(ethnicity_logits, dim=1)
        gender_probs = F.softmax(gender_logits, dim=1)

        # Extract integer class IDs from tensors
        age_class_id = torch.argmax(age_probs, dim=1).item()
        ethnicity_class_id = torch.argmax(ethnicity_probs, dim=1).item()
        gender_class_id = torch.argmax(gender_probs, dim=1).item()

        # Extract probabilities
        age_probability = torch.max(age_probs, dim=1)[0].item()
        ethnicity_probability = torch.max(ethnicity_probs, dim=1)[0].item()
        gender_probability = torch.max(gender_probs, dim=1)[0].item()

        # Append the results
        age_result = Result("Age", age_probability, AGES[age_class_id])
        results.append(age_result)

        ethnicity_result = Result(
            "Ethnicity", ethnicity_probability, ETHNICITIES[ethnicity_class_id]
        )
        results.append(ethnicity_result)

        gender_result = Result("Gender", gender_probability, GENDERS[gender_class_id])
        results.append(gender_result)

    return results
