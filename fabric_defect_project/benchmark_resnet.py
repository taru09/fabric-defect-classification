
from pathlib import Path
import time
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

BASE = Path(__file__).resolve().parent
TEST = BASE / "test"
MODEL_FILE = BASE / "resnet18_fabric.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(
    torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
)
model.eval()

test_data = datasets.ImageFolder(TEST)
image_paths = [p for p, label in test_data.samples]

# Warm up the model
sample = transform(
    Image.open(image_paths[0]).convert("RGB")
).unsqueeze(0)

with torch.no_grad():
    for _ in range(5):
        model(sample)

# Measure all 509 predictions
start = time.perf_counter()

with torch.no_grad():
    for path in image_paths:
        image = transform(
            Image.open(path).convert("RGB")
        ).unsqueeze(0)
        model(image)

elapsed = time.perf_counter() - start

print("Images tested:", len(image_paths))
print(f"Total inference time: {elapsed:.3f} seconds")
print(f"Average per image: {elapsed / len(image_paths) * 1000:.2f} ms")
