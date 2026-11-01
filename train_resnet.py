
from pathlib import Path
import csv
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

# --------------------------------
# 1. SETTINGS
# --------------------------------
BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
VALID_DIR = BASE_DIR / "valid"
TEST_DIR = BASE_DIR / "test"

DINO_CSV = BASE_DIR / "predictions.csv"
OUTPUT_CSV = BASE_DIR / "resnet18_predictions.csv"
MODEL_FILE = BASE_DIR / "resnet18_fabric.pth"

DEVICE = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS = 5

print("Training device:", DEVICE)

# --------------------------------
# 2. IMAGE TRANSFORMS
# --------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------
# 3. LOAD DATASET
# --------------------------------
train_data = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform
)
valid_data = datasets.ImageFolder(
    VALID_DIR, transform=valid_transform
)

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(valid_data))

if train_data.classes != ["defect_free", "minor_defect"]:
    raise ValueError("Please check your class folder names.")

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)

valid_loader = DataLoader(
    valid_data, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# --------------------------------
# 4. LOAD RESNET18
# --------------------------------
print("\nLoading pretrained ResNet18...")

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze earlier layers to make CPU training faster
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final ResNet block and classification layer
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(DEVICE)

# Handle class imbalance
class_counts = torch.bincount(
    torch.tensor(train_data.targets)
).float()

class_weights = len(train_data) / (2 * class_counts)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# --------------------------------
# 5. TRAIN MODEL
# --------------------------------
best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_data)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions.cpu() == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_weights = copy.deepcopy(model.state_dict())

# Save best model
model.load_state_dict(best_weights)
torch.save(model.state_dict(), MODEL_FILE)

print("\nTraining complete!")
print("Best validation accuracy:", round(best_accuracy, 2), "%")
print("Model saved:", MODEL_FILE)

# --------------------------------
# 6. EVALUATE SAME 50 DINOv3 IMAGES
# --------------------------------
if not DINO_CSV.exists():
    raise FileNotFoundError("predictions.csv not found.")

model.eval()
results = []

with open(DINO_CSV, newline="", encoding="utf-8") as file:
    dino_rows = list(csv.DictReader(file))

if len(dino_rows) != 50:
    raise ValueError(
        f"Expected 50 DINOv3 images, found {len(dino_rows)}."
    )

correct = 0

for row in dino_rows:
    filename = row["filename"]
    actual_class = row["actual_class"]

    image_path = TEST_DIR / actual_class / filename

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = valid_transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_index = probabilities.argmax().item()
    predicted_class = train_data.classes[predicted_index]
    confidence = probabilities[predicted_index].item()

    is_correct = predicted_class == actual_class
    correct += int(is_correct)

    results.append({
        "filename": filename,
        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "correct": is_correct
    })

# --------------------------------
# 7. SAVE PREDICTIONS
# --------------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "actual_class",
            "predicted_class",
            "confidence",
            "correct"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nRESNET18 TEST RESULTS")
print("---------------------")
print("Images evaluated:", len(results))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(results) - correct)
print("Accuracy:", round(correct / len(results) * 100, 2), "%")
print("Predictions saved:", OUTPUT_CSV)
