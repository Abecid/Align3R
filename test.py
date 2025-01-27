import os
import numpy as np

path = "/scratch/partial_datasets/align3r/data/spring_proc/train/0001"

pfm_file = "0248_depth.pfm"
npz_file = "0248_rgb_pred_depth_moge.npz"

# Function to read PFM files
def read_pfm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().decode().strip()
        dimensions = f.readline().decode().strip()
        scale = f.readline().decode().strip()
        print(f"PFM Header: {header}, Dimensions: {dimensions}, Scale: {scale}")
        # For brevity, skipping actual data parsing here.

# Read and summarize the PFM file
read_pfm(os.path.join(path, pfm_file))

# Read and summarize the NPZ file
npz_path = os.path.join(path, npz_file)
data = np.load(npz_path)
print(f"Contents of {npz_file}:")
for key, value in data.items():
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
