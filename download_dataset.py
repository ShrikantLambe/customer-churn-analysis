"""
Script to download Telco Customer Churn dataset using kagglehub
"""
import kagglehub
import shutil
import os

# Download latest version
download_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Path to dataset files:", download_path)

# Move the CSV to the data directory
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Find the CSV file in the downloaded folder
for file in os.listdir(download_path):
    if file.endswith(".csv"):
        src = os.path.join(download_path, file)
        dst = os.path.join(data_dir, file)
        shutil.move(src, dst)
        print(f"Moved {file} to {data_dir}/")
        break
else:
    print("No CSV file found in the downloaded dataset.")
