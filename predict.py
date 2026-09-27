import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import sys

# Device (Mac MPS support)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("dr_model.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Get image path from command line
image_path = sys.argv[1]
x
# Open image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

class_names = ["DR", "No_DR"]

print(f"Prediction: {class_names[predicted.item()]} (Confidence: {confidence.item()*100:.2f}%)")

