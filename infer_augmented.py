
from pathlib import Path
import csv
import time
from getpass import getpass
from inference_sdk import InferenceHTTPClient

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
OUTPUT_FILE = BASE_DIR / "roboflow_original_full_predictions.csv"

MODEL_ID = "prashansas-workspace/fabric-detection-yhlkp-1-vit_small_patch16_dinov3.lvd1689m-t1"
API_KEY = getpass("Enter your Roboflow Private API Key: ")

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

rows = []
total_time = 0
successful = 0

for actual_class in ["defect_free", "minor_defect"]:

    class_folder = TEST_DIR / actual_class

    if not class_folder.exists():
        raise FileNotFoundError(f"Folder missing: {class_folder}")

    images = sorted([
        p for p in class_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    ])

    print(f"\nProcessing {actual_class}: {len(images)} images")

    for image_path in images:

        print(f"Predicting {successful + 1}: {image_path.name}")

        start = time.perf_counter()

        result = client.infer(
            str(image_path),
            model_id=MODEL_ID
        )

        elapsed = time.perf_counter() - start
        total_time += elapsed

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
            "correct": predicted_class == actual_class,
            "api_request_seconds": round(elapsed, 4)
        })

        successful += 1

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "filename",
            "actual_class",
            "predicted_class",
            "confidence",
            "correct",
            "api_request_seconds"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

correct = sum(row["correct"] for row in rows)
accuracy = correct / len(rows) * 100

print("\nAUGMENTED DINOv3 RESULTS")
print("------------------------")
print("Images evaluated:", len(rows))
print("Correct predictions:", correct)
print("Incorrect predictions:", len(rows) - correct)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Total API request time: {total_time:.2f} seconds")
print(f"Average API request time: {total_time / len(rows) * 1000:.2f} ms")
print("CSV saved:", OUTPUT_FILE)
