
import os
import argparse
import subprocess
import sys
import shutil

def run_command(command, cwd=None, env=None):
    print(f"\n[PIPELINE] Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[PIPELINE] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run PlanarGS Pipeline")
    parser.add_argument("--source_path", "-s", required=True, help="Path to the dataset directory")
    parser.add_argument("--model_path", "-m", default=None, help="Output model path")
    parser.add_argument("--text_prompts", "-t", required=True, help="Text prompts for LP3 (e.g., 'wall. floor. window.')")
    parser.add_argument("--skip_geomprior", action="store_true", help="Skip geometric prior step")
    parser.add_argument("--skip_lp3", action="store_true", help="Skip LP3 step")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    parser.add_argument("--skip_render", action="store_true", help="Skip rendering step")
    parser.add_argument("--image_size", type=int, default=512, help="Inference size for DUSt3R (default: 512)")
    parser.add_argument("--group_size", type=int, default=10, help="Number of images per group (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DUSt3R inference (default: 8)")
    
    # Catch-all for extra arguments to pass to train.py
    args, unknown_args = parser.parse_known_args()

    source_path = os.path.abspath(args.source_path)
    if not os.path.exists(source_path):
        print(f"[PIPELINE] Error: Source path does not exist: {source_path}")
        sys.exit(1)

    # Determine default model path
    if args.model_path:
        model_path = os.path.abspath(args.model_path)
    else:
        basename = os.path.basename(source_path.rstrip(os.sep))
        # Default to output/<basename> in the current working directory (project root)
        model_path = os.path.abspath(f"output/{basename}")

    print(f"[PIPELINE] Source: {source_path}")
    print(f"[PIPELINE] Output: {model_path}")

    # Environment Setup
    # Add project root to PYTHONPATH so submodules can be found
    project_root = os.getcwd()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # --- Dataset Preparation (Auto-Symlinking) ---
    sparse_path = os.path.join(source_path, "sparse")
    sfm_colmap_path = os.path.join(source_path, "sfm_colmap")
    
    # Check if we need to link sfm_colmap -> sparse
    if not os.path.exists(sparse_path) and os.path.exists(sfm_colmap_path):
        print("[PIPELINE] 'sparse' directory missing, but 'sfm_colmap' found. Attempting to link...")
        
        # Check for 'sparse' inside 'sfm_colmap'
        target = None
        # Priority 1: sfm_colmap/sparse/0 (Mutagen style)
        candidate_1 = os.path.join(sfm_colmap_path, "sparse", "0")
        # Priority 2: sfm_colmap/sparse
        candidate_2 = os.path.join(sfm_colmap_path, "sparse")
        
        if os.path.exists(candidate_1) and os.path.isdir(candidate_1):
             target = "sfm_colmap/sparse/0"
        elif os.path.exists(candidate_2) and os.path.isdir(candidate_2):
             target = "sfm_colmap/sparse"
        
        if target:
            try:
                # Create symlink relative to source_path
                # os.symlink requires absolute paths or careful relative handling. 
                # Simplest is to run ln -s in the directory.
                cmd = f"ln -sf {target} sparse"
                print(f"[PIPELINE] Linking: {cmd}")
                run_command(cmd, cwd=source_path)
            except Exception as e:
                print(f"[PIPELINE] Warning: Failed to create symlink: {e}")
        else:
             print("[PIPELINE] Warning: 'sfm_colmap' found but could not locate valid sparse directory structure inside it.")

    # --- Pipelines ---
    
    # 1. Geometric Prior (DUSt3R)
    if not args.skip_geomprior:
        print("\n=== Running Geometric Prior (DUSt3R) ===")
        # Note: --ckpt_mv path needs to be absolute or relative to where we run run_geomprior.
        # Since we run run_geomprior in source_path (usually), or project_root? 
        # The original pixi task ran python run_geomprior.py -s . inside the data dir. 
        # But run_geomprior.py is in project root.
        # Let's run everything from project root and pass absolute path to -s.
        
        ckpt_path = os.path.join(project_root, "ckpt", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        cmd = f"python run_geomprior.py -s {source_path} --group_size {args.group_size} --ckpt_mv {ckpt_path} --image_size {args.image_size} --batch_size {args.batch_size}"
        run_command(cmd, cwd=project_root, env=env)

    # 2. LP3 (Language-Guided Plane Priors)
    if not args.skip_lp3:
        print("\n=== Running LP3 ===")
        # Need to handle text prompts. If spaces are involved, quote them.
        cmd = f"python run_lp3.py -s {source_path} -t '{args.text_prompts}'"
        run_command(cmd, cwd=project_root, env=env)

    # 3. Training
    if not args.skip_train:
        print("\n=== Running Training ===")
        # Pass unknown args to train.py
        extra_args = " ".join(unknown_args)
        cmd = f"python train.py -s {source_path} -m {model_path} {extra_args}"
        run_command(cmd, cwd=project_root, env=env)

    # 4. Rendering
    if not args.skip_render:
        print("\n=== Running Rendering ===")
        cmd = f"python render.py -m {model_path}"
        run_command(cmd, cwd=project_root, env=env)

    print("\n[PIPELINE] Finished successfully.")

if __name__ == "__main__":
    main()
