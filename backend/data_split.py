import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# === PATHS ===
reports_csv = Path(r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\indiana_reports.csv")
projections_csv = Path(r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\indiana_projections.csv")
images_dir = Path(r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\images\images_normalized")

# Output folders
output_dir = Path(r"C:\Users\Kunal Gulati\OneDrive\Desktop\minor project\backend\output_images")
train_dir = output_dir / "train"
test_dir = output_dir / "test"

# Create output folders if they don't exist
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# === READ THE CSVs ===
reports_df = pd.read_csv(reports_csv)
projections_df = pd.read_csv(projections_csv)

# === GET UNIQUE UIDs AND SPLIT ===
uids = reports_df['uid'].unique()
train_uids, test_uids = train_test_split(uids, test_size=0.2, random_state=42)

# === FUNCTION TO COPY IMAGES ===
def copy_images(uids, target_folder):
    count = 0
    for uid in uids:
        # Look for all images starting with uid_
        matched_files = list(images_dir.glob(f"{uid}_*.png"))
        for src in matched_files:
            dst = target_folder / src.name
            shutil.copy2(src, dst)  # change to shutil.move if you want to move
            count += 1
    return count

# Copy the images
train_count = copy_images(train_uids, train_dir)
test_count = copy_images(test_uids, test_dir)

# === SPLIT CSVs ACCORDING TO UIDs ===
train_reports = reports_df[reports_df['uid'].isin(train_uids)]
test_reports = reports_df[reports_df['uid'].isin(test_uids)]

train_projections = projections_df[projections_df['uid'].isin(train_uids)]
test_projections = projections_df[projections_df['uid'].isin(test_uids)]

# Save the splits
train_reports.to_csv(output_dir / "train_reports.csv", index=False)
test_reports.to_csv(output_dir / "test_reports.csv", index=False)
train_projections.to_csv(output_dir / "train_projections.csv", index=False)
test_projections.to_csv(output_dir / "test_projections.csv", index=False)

print(f"✅ Done!")
print(f"Train images: {train_count}, Test images: {test_count}")
print(f"Train reports: {len(train_reports)}, Test reports: {len(test_reports)}")
print(f"Train projections: {len(train_projections)}, Test projections: {len(test_projections)}")
