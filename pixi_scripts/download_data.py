from huggingface_hub import snapshot_download
import os

print("Downloading dataset...")
snapshot_download(
    repo_id="Isabellaxr/PlanarGS_dataset",
    repo_type="dataset",
    allow_patterns="PlanarGS_dataset/replica/room0/**",
    local_dir="hf_download",
    resume_download=True
)
print("Download complete.")
