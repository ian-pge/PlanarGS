#!/bin/bash
set -e

# Force CUDA_HOME to be the current prefix if not set
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
# Inject compatibility header for FLT_MAX
export NVCC_APPEND_FLAGS="--pre-include /workspace/PlanarGS/pixi_scripts/fix_compat.h $NVCC_APPEND_FLAGS"
echo "CUDA_HOME is set to: $CUDA_HOME"

if pip show simple_knn > /dev/null 2>&1; then
    echo "simple_knn is already installed. Skipping..."
else
    echo "Installing simple-knn..."
    pip install -e submodules/simple-knn --no-build-isolation
fi

# echo "Installing pytorch3d..."
# pip install -e submodules/pytorch3d --no-build-isolation

if pip show diff_plane_rasterization > /dev/null 2>&1; then
    echo "diff-plane-rasterization is already installed. Skipping..."
else
    echo "Installing diff-plane-rasterization..."
    pip install submodules/diff-plane-rasterization --no-build-isolation
fi

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

if pip show segment-anything > /dev/null 2>&1; then
    echo "segment-anything is already installed. Skipping..."
else
    pip install -e segment_anything
fi

# Install GroundingDINO
if [ -d "GroundingDINO" ]; then
    if pip show groundingdino > /dev/null 2>&1; then
        echo "groundingdino is already installed. Skipping..."
    else
        echo "Installing GroundingDINO..."
        pip install --no-build-isolation -e GroundingDINO
    fi
else
    echo "GroundingDINO directory not found inside groundedsam!"
fi
cd ../..

echo "Submodule installation complete."
