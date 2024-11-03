# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes):
    """Creates the MobileNetV2 model with adjusted classifier."""
    model = models.mobilenet_v2(weights='DEFAULT')  # Load pre-trained MobileNetV2
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # Adjust the classifier layer
    return model

def load_model(model_path: str, class_names: list):
    """Loads the trained model and returns it."""
    num_classes = len(class_names)
    model = create_model(num_classes)
    model.load_state_dict(torch.load(model_path))  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model
