#!/bin/bash
set -e

# Force CUDA_HOME to be the current prefix if not set
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
# Inject compatibility header for FLT_MAX
export NVCC_APPEND_FLAGS="--pre-include /workspace/PlanarGS/fix_compat.h $NVCC_APPEND_FLAGS"
echo "CUDA_HOME is set to: $CUDA_HOME"

echo "Installing simple-knn..."
pip install -e submodules/simple-knn --no-build-isolation

# echo "Installing pytorch3d..."
# pip install -e submodules/pytorch3d --no-build-isolation

echo "Installing diff-plane-rasterization..."
pip install submodules/diff-plane-rasterization --no-build-isolation

# GroundedSAM
if [ ! -d "submodules/groundedsam" ]; then
    echo "Cloning GroundedSAM..."
    cd submodules
    git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
    mv Grounded-Segment-Anything groundedsam
    cd ..
fi

echo "Installing GroundedSAM components..."
cd submodules/groundedsam
pip install -e segment_anything
# Install GroundingDINO
if [ -d "GroundingDINO" ]; then
    echo "Installing GroundingDINO..."
    pip install --no-build-isolation -e GroundingDINO
else
    echo "GroundingDINO directory not found inside groundedsam!"
fi
cd ../..

echo "Submodule installation complete."
