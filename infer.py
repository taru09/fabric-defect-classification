
from pathlib import Path
import csv
import random
from getpass import getpass
from inference_sdk import InferenceHTTPClient

# Your trained Roboflow model
MODEL_ID = "prashansas-workspace/fabric-detection-yhlkp-1-vit_small_patch16_dinov3.lvd1689m-t1"

# Finds the test folder beside this Python file
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
OUTPUT_FILE = BASE_DIR / "predictions.csv"

# Enter your private API key when prompted
API_KEY = getpass("Enter your Roboflow Private API Key: ")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
randomizer = random.Random(42)

rows = []

# Select 25 images from each class (50 total)
for actual_class in ["defect_free", "minor_defect"]:
    class_folder = TEST_DIR / actual_class

    if not class_folder.exists():
        raise FileNotFoundError(f"Folder not found: {class_folder}")

    images = [
        p for p in class_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ]

    if len(images) < 25:
        raise ValueError(
            f"Only {len(images)} images found in {class_folder}; need 25."
        )

    selected_images = randomizer.sample(images, 25)

    for image_path in selected_images:
        print(f"Predicting: {image_path.name}")

        result = client.infer(str(image_path), model_id=MODEL_ID)
        predictions = result.get("predictions", [])

        if predictions:
            best = max(
                predictions,
                key=lambda item: item.get("confidence", 0)
            )
            predicted_class = best.get("class", "")
            confidence = best.get("confidence", "")
        else:
            predicted_class = result.get("top", "")
            confidence = result.get("confidence", "")

        rows.append({
            "filename": image_path.name,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "correct": predicted_class == actual_class
        })

# Save the results
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
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
    writer.writerows(rows)

correct_count = sum(row["correct"] for row in rows)

print("\nFinished!")
print(f"Images evaluated: {len(rows)}")
print(f"Correct predictions: {correct_count}")
print(f"Accuracy: {correct_count / len(rows) * 100:.2f}%")
print(f"CSV saved to: {OUTPUT_FILE}")
