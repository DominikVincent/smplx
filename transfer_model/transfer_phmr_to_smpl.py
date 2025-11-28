""" this file takes the SMPLX output of the PHMR model and transfers it to SMPL format.

The phmr output is saved as npz files with the following keys:
['codecVersion', 'smplVersion', 'gender', 'shapeParameters', 'frameCount', 'frameRate', 'bodyTranslation', 'bodyPose']

it can be loaded like this:

    data = np.load(smpl_file_path, allow_pickle=True)


# demo smplx data is here: /home/dominik/Documents/repos/PromptHMR/results/TC_S1_acting1_cam4_20251127_170844/subject-1.smpl

the script should output smpl files with the following keys:
pose: (N, 72) array of body pose parameters
betas: (10,) array of shape parameters
trans: (N, 3) array of global translation parameters
"""

import sys
import os
import os.path as osp
import argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory to path to allow imports
repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import smplx
from transfer_model.transfer_model import run_fitting
from transfer_model.utils import read_deformation_transfer, batch_rot2aa

def main():
    parser = argparse.ArgumentParser(description='Transfer PHMR SMPL-X data to SMPL')
    parser.add_argument('--input-path', type=str, required=True, help='Path to PHMR smpl file')
    parser.add_argument('--output-path', type=str, required=True, help='Path to output smpl file')
    parser.add_argument('--model-folder', type=str, default='/home/dominik/Documents/repos/AiOS/data/body_models', help='Path to body models')
    parser.add_argument('--exp-cfg', type=str, default='config_files/smplx2smpl.yaml', help='Config file')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for fitting')
    args = parser.parse_args()

    # Load config
    if not osp.exists(args.exp_cfg):
        # Try to find it relative to the script or repo root
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
        alt_cfg = osp.join(repo_root, args.exp_cfg)
        if osp.exists(alt_cfg):
            args.exp_cfg = alt_cfg
        else:
            print(f"Config file not found: {args.exp_cfg}")
            sys.exit(1)

    exp_cfg = OmegaConf.load(args.exp_cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load PHMR data
    print(f"Loading data from {args.input_path}")
    data = np.load(args.input_path, allow_pickle=True)
    # keys: ['codecVersion', 'smplVersion', 'gender', 'shapeParameters', 'frameCount', 'frameRate', 'bodyTranslation', 'bodyPose']
    
    # Determine gender
    gender_str = str(data['gender'])
    if 'female' in gender_str.lower():
        gender = 'female'
    elif 'male' in gender_str.lower():
        gender = 'male'
    else:
        gender = 'neutral'
    print(f"Gender: {gender}")

    # Load SMPL-X model (Source)
    print("Loading SMPL-X model...")
    smplx_model = smplx.create(
        args.model_folder, 
        model_type='smplx',
        gender=gender, 
        use_pca=False,
        num_betas=10,
        batch_size=args.batch_size
    ).to(device)

    # Load SMPL model (Target)
    print("Loading SMPL model (Target)...")
    target_model = smplx.create(
        args.model_folder,
        model_type=exp_cfg.body_model.model_type,
        gender=exp_cfg.body_model.gender,
        num_betas=exp_cfg.body_model.smpl.betas.num,
        batch_size=args.batch_size
    ).to(device)

    # Load deformation transfer
    print("Loading deformation transfer matrix...")
    def_path = exp_cfg.deformation_transfer_path
    if not osp.exists(def_path):
         repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
         def_path = osp.join(repo_root, def_path)
    
    def_matrix = read_deformation_transfer(def_path, device=device)

    # Process data
    num_frames = int(data['frameCount'])
    print(f"Processing {num_frames} frames...")
    
    # Extract parameters
    betas = torch.tensor(data['shapeParameters'], dtype=torch.float32).to(device) # (10,)
    body_pose_phmr = torch.tensor(data['bodyPose'], dtype=torch.float32).to(device) # (N, 22, 3)
    transl = torch.tensor(data['bodyTranslation'], dtype=torch.float32).to(device) # (N, 3)

    # Split body_pose_phmr into global_orient and body_pose
    global_orient = body_pose_phmr[:, 0:1] # (N, 1, 3)
    body_pose = body_pose_phmr[:, 1:] # (N, 21, 3)
    
    # Expand betas
    betas_batch = betas.unsqueeze(0).expand(num_frames, -1)

    # Output storage
    out_pose = []
    out_betas = []
    out_trans = []

    for i in tqdm(range(0, num_frames, args.batch_size)):
        curr_batch_size = min(args.batch_size, num_frames - i)
        
        batch_betas = betas_batch[i:i+curr_batch_size]
        batch_global_orient = global_orient[i:i+curr_batch_size]
        batch_body_pose = body_pose[i:i+curr_batch_size]
        batch_transl = transl[i:i+curr_batch_size]

        # Create zero tensors for other SMPL-X parameters to handle variable batch size
        batch_jaw_pose = torch.zeros((curr_batch_size, 1, 3), dtype=torch.float32, device=device)
        batch_leye_pose = torch.zeros((curr_batch_size, 1, 3), dtype=torch.float32, device=device)
        batch_reye_pose = torch.zeros((curr_batch_size, 1, 3), dtype=torch.float32, device=device)
        batch_left_hand_pose = torch.zeros((curr_batch_size, 15, 3), dtype=torch.float32, device=device)
        batch_right_hand_pose = torch.zeros((curr_batch_size, 15, 3), dtype=torch.float32, device=device)
        batch_expression = torch.zeros((curr_batch_size, 10), dtype=torch.float32, device=device)

        # Generate Source Vertices (SMPL-X)
        with torch.no_grad():
            # Adjust batch size of model if needed (smplx handles it usually, but let's be safe)
            # smplx forward handles batch size of inputs.
            smplx_output = smplx_model(
                betas=batch_betas,
                global_orient=batch_global_orient,
                body_pose=batch_body_pose,
                transl=batch_transl,
                jaw_pose=batch_jaw_pose,
                leye_pose=batch_leye_pose,
                reye_pose=batch_reye_pose,
                left_hand_pose=batch_left_hand_pose,
                right_hand_pose=batch_right_hand_pose,
                expression=batch_expression,
                return_verts=True
            )
            source_vertices = smplx_output.vertices
            source_faces = smplx_model.faces_tensor

        # Prepare batch for fitting
        fit_batch = {
            'vertices': source_vertices,
            'faces': source_faces
        }

        # Run fitting
        var_dict = run_fitting(
            exp_cfg,
            fit_batch,
            target_model,
            def_matrix,
            mask_ids=None
        )

        # Collect results
        res_body_pose = var_dict['body_pose'].detach().cpu().numpy() 
        res_global_orient = var_dict['global_orient'].detach().cpu().numpy() 
        res_betas = var_dict['betas'].detach().cpu().numpy() # (B, 10)
        res_transl = var_dict['transl'].detach().cpu().numpy() # (B, 3)
        
        # Combine pose
        res_pose = np.concatenate([res_global_orient, res_body_pose], axis=1) # (B, 24, 3, 3)
        
        # convert to axis angle shape B, 24, 3
        axis_angles = batch_rot2aa(torch.tensor(res_pose).view(-1, 3, 3)).reshape(-1, 24, 3).numpy() # (B, 24, 3)

        out_pose.append(axis_angles)
        out_betas.append(res_betas)
        out_trans.append(res_transl)

    # Concatenate all
    final_pose = np.concatenate(out_pose, axis=0)
    final_betas = np.mean(np.concatenate(out_betas, axis=0), axis=0) 
    final_trans = np.concatenate(out_trans, axis=0)

    # Save
    os.makedirs(osp.dirname(args.output_path), exist_ok=True)
    np.savez(args.output_path, pose=final_pose, betas=final_betas, trans=final_trans)
    print(f"Saved to {args.output_path}.npz")

if __name__ == '__main__':
    main()


