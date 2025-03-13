"""
Inference script for applying super-resolution to medical images.
Supports 3D medical images in NIfTI (.nii) and MHA formats.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.generator import Generator
from utils import calculate_psnr, calculate_ssim
from config import DATA

def load_model(checkpoint_path, device):
    """
    Load a trained generator model.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model to
        
    Returns:
        nn.Module: Loaded generator model
    """
    generator = Generator().to(device)
    
    # Load checkpoint
    if checkpoint_path.endswith('.pth'):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both full SRGAN checkpoint and generator-only checkpoint
        if 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    
    generator.eval()
    return generator

def load_volume(file_path, file_extension='.nii'):
    """
    Load a 3D medical image volume.
    
    Args:
        file_path (str): Path to the medical image file
        file_extension (str): File extension ('.nii' or '.mha')
        
    Returns:
        tuple: (volume data as numpy array, original image object for saving later)
    """
    if file_extension == '.nii':
        # Load NIfTI file
        nifti_img = nib.load(file_path)
        volume = nifti_img.get_fdata()
        return volume, nifti_img
    elif file_extension == '.mha':
        # Load MHA file
        itk_img = sitk.ReadImage(file_path)
        volume = sitk.GetArrayFromImage(itk_img)
        # Convert from [slices, height, width] to [height, width, slices]
        volume = np.transpose(volume, (1, 2, 0))
        return volume, itk_img
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def save_volume(volume, original_img, output_path, file_extension='.nii'):
    """
    Save a 3D medical image volume.
    
    Args:
        volume (numpy.ndarray): Volume data to save
        original_img: Original image object (nib.Nifti1Image or SimpleITK.Image)
        output_path (str): Path to save the output file
        file_extension (str): File extension ('.nii' or '.mha')
    """
    if file_extension == '.nii':
        # Create a new NIfTI image with the same header as the original
        output_img = nib.Nifti1Image(volume, original_img.affine, header=original_img.header)
        nib.save(output_img, output_path)
    elif file_extension == '.mha':
        # Convert from [height, width, slices] to [slices, height, width]
        volume = np.transpose(volume, (2, 0, 1))
        output_img = sitk.GetImageFromArray(volume)
        
        # Copy metadata from original image
        for key in original_img.GetMetaDataKeys():
            output_img.SetMetaData(key, original_img.GetMetaData(key))
        
        # Copy spacing, origin, and direction from original image
        output_img.SetSpacing(original_img.GetSpacing())
        output_img.SetOrigin(original_img.GetOrigin())
        output_img.SetDirection(original_img.GetDirection())
        
        sitk.WriteImage(output_img, output_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

def run_inference(args):
    """
    Run inference on a medical image volume.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    generator = load_model(args.checkpoint, device)
    
    # Load input volume
    print(f"Loading input volume from {args.input}")
    volume, original_img = load_volume(args.input, args.file_extension)
    
    # Get volume dimensions
    h, w, d = volume.shape
    print(f"Input volume shape: {volume.shape}")
    
    # Create output volume
    sr_volume = np.zeros((h * DATA['scale_factor'], w * DATA['scale_factor'], d), dtype=np.float32)
    
    # Process each slice
    print("Processing slices...")
    start_time = time.time()
    psnr_values = []
    ssim_values = []
    
    for slice_idx in tqdm(range(d)):
        # Extract slice
        lr_slice = volume[:, :, slice_idx]
        
        # Skip if slice is all zeros or very sparse
        if lr_slice.max() <= DATA['slice_threshold']:
            # Just upscale with bicubic for empty slices
            sr_slice = F.interpolate(
                torch.from_numpy(lr_slice).float().unsqueeze(0).unsqueeze(0),
                scale_factor=DATA['scale_factor'],
                mode='bicubic',
                align_corners=False
            ).squeeze().cpu().numpy()
            sr_volume[:, :, slice_idx] = sr_slice
            continue
        
        # Convert to tensor
        lr_tensor = torch.from_numpy(lr_slice).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # Normalize to [0, 1] if needed
        normalized = False
        lr_min = 0
        lr_max = 1

        if lr_tensor.min() < 0 or lr_tensor.max() > 1:
            lr_min = lr_tensor.min()
            lr_max = lr_tensor.max()
            lr_tensor = (lr_tensor - lr_min) / (lr_max - lr_min)
            normalized = True
        
        # Resize if needed
        if lr_tensor.shape[2] != DATA['lr_size'] or lr_tensor.shape[3] != DATA['lr_size']:
            lr_tensor = F.interpolate(
                lr_tensor,
                size=(DATA['lr_size'], DATA['lr_size']),
                mode='bicubic',
                align_corners=False
            )
        
        # Generate super-resolution slice
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)
        
        # Convert back to original size if needed
        if sr_tensor.shape[2] != h * DATA['scale_factor'] or sr_tensor.shape[3] != w * DATA['scale_factor']:
            sr_tensor = F.interpolate(
                sr_tensor,
                size=(h * DATA['scale_factor'], w * DATA['scale_factor']),
                mode='bicubic',
                align_corners=False
            )
        
        # Convert back to original range if normalized
        if normalized:
            sr_tensor = sr_tensor * (lr_max - lr_min) + lr_min
        
        # Convert to numpy and store in output volume
        sr_slice = sr_tensor.squeeze().cpu().numpy()
        sr_volume[:, :, slice_idx] = sr_slice
        
        # Calculate metrics for non-empty slices if ground truth is available
        if args.ground_truth:
            gt_volume, _ = load_volume(args.ground_truth, args.file_extension)
            gt_slice = gt_volume[:, :, slice_idx]
            
            # Only calculate metrics if ground truth slice is not empty
            if gt_slice.max() > DATA['slice_threshold']:
                psnr = calculate_psnr(sr_slice, gt_slice)
                ssim = calculate_ssim(sr_slice, gt_slice)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    
    # Report processing time
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    
    # Report metrics if ground truth was provided
    if args.ground_truth and psnr_values:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save output volume
    print(f"Saving output volume to {args.output}")
    save_volume(sr_volume, original_img, args.output, args.file_extension)
    
    # Visualize sample slices if requested
    if args.visualize:
        middle_slice = d // 2
        
        # Find a non-empty slice near the middle if the middle slice is empty
        if volume[:, :, middle_slice].max() <= DATA['slice_threshold']:
            for offset in range(1, d // 2):
                if middle_slice + offset < d and volume[:, :, middle_slice + offset].max() > DATA['slice_threshold']:
                    middle_slice += offset
                    break
                elif middle_slice - offset >= 0 and volume[:, :, middle_slice - offset].max() > DATA['slice_threshold']:
                    middle_slice -= offset
                    break
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Low resolution
        plt.subplot(1, 2, 1)
        plt.imshow(volume[:, :, middle_slice], cmap='gray')
        plt.title(f'Low-Resolution (Slice {middle_slice})')
        plt.axis('off')
        
        # Super resolution
        plt.subplot(1, 2, 2)
        plt.imshow(sr_volume[:, :, middle_slice], cmap='gray')
        plt.title(f'Super-Resolution (Slice {middle_slice})')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or show visualization
        vis_path = os.path.splitext(args.output)[0] + '_visualization.png'
        plt.savefig(vis_path)
        print(f"Visualization saved to {vis_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply super-resolution to medical images')
    parser.add_argument('--input', type=str, required=True, help='Input medical image file')
    parser.add_argument('--output', type=str, required=True, help='Output medical image file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--file-extension', type=str, default='.nii', choices=['.nii', '.mha'], 
                        help='File extension of input/output files')
    parser.add_argument('--ground-truth', type=str, default=None, help='Ground truth high-resolution image for evaluation')
    parser.add_argument('--visualize', action='store_true', help='Create visualization of results')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Check if checkpoint file exists
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    run_inference(args) 