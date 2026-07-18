import os
import shutil
import cv2
import pandas as pd

RAW_IMGS = "./raw_data/images"
RAW_LABELS = "./raw_data/labels"
OUT_IMGS = "./training_set/images"
OUT_LABELS = "./training_set/labels"

for folder in [OUT_IMGS, OUT_LABELS]:
    os.makedirs(folder, exist_ok=True)

stats = []

print("Firing up OpenCV to clean the dataset...")

for filename in os.listdir(RAW_IMGS):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    name_only = os.path.splitext(filename)[0]
    img_path = os.path.join(RAW_IMGS, filename)
    label_path = os.path.join(RAW_LABELS, f"{name_only}.txt")

    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        stats.append({"file": filename, "status": "dropped_no_label"})
        continue

    img = cv2.imread(img_path)
    if img is None:
        stats.append({"file": filename, "status": "dropped_corrupted"})
        continue

    resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)

    cv2.imwrite(os.path.join(OUT_IMGS, f"{name_only}.jpg"), resized)
    shutil.copy(label_path, os.path.join(OUT_LABELS, f"{name_only}.txt"))

    stats.append({"file": filename, "status": "success"})

df = pd.DataFrame(stats)

print("\nAll done! Here is the breakdown:")
print(df['status'].value_counts().to_string())
log_path = "./training_set/cleaning_report.csv"
df.to_csv(log_path, index=False)
print(f"\nSaved detailed log to: {log_path}")