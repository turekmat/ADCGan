"""
Utility functions for the Super-Resolution GAN model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from config import DATA, DIRS

def create_dirs():
    """
    Create necessary directories if they don't exist.
    """
    for dir_name in DIRS.values():
        os.makedirs(dir_name, exist_ok=True)

def downsample_tensor(x, scale_factor=DATA['scale_factor']):
    """
    Downsample tensor by a scale factor.
    Uses bicubic interpolation without introducing artifacts.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        scale_factor (int): Downsampling scale factor
        
    Returns:
        torch.Tensor: Downsampled tensor
    """
    _, _, h, w = x.size()
    new_h, new_w = h // scale_factor, w // scale_factor
    return F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)

def upsample_tensor(x, scale_factor=DATA['scale_factor']):
    """
    Upsample tensor by a scale factor.
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        scale_factor (int): Upsampling scale factor
        
    Returns:
        torch.Tensor: Upsampled tensor
    """
    _, _, h, w = x.size()
    new_h, new_w = h * scale_factor, w * scale_factor
    return F.interpolate(x, size=(new_h, new_w), mode='bicubic', align_corners=False)

def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        
    Returns:
        float: PSNR value
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    # Ensure images are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        
    Returns:
        float: SSIM value
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Ensure images are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # For multi-channel images
    if img1.ndim == 4:  # [B, C, H, W]
        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_values.append(
                structural_similarity(
                    img1[i, 0], img2[i, 0], data_range=1.0, multichannel=False
                )
            )
        return np.mean(ssim_values)
    else:  # For single image
        return structural_similarity(img1, img2, data_range=1.0, multichannel=False)

def save_image(tensor, filename, nrow=4):
    """
    Save a batch of images to a file.
    
    Args:
        tensor (torch.Tensor): Tensor of shape [B, C, H, W]
        filename (str): Output filename
        nrow (int): Number of images per row
    """
    from torchvision.utils import save_image as _save_image
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    # Save image
    _save_image(tensor, filename, nrow=nrow, normalize=False)

def visualize_results(lr_img, sr_img, hr_img, epoch, batch_idx=0, save_path=None):
    """
    Visualize low-resolution, super-resolution, and high-resolution images.
    
    Args:
        lr_img (torch.Tensor): Low-resolution image tensor [B, C, H, W]
        sr_img (torch.Tensor): Super-resolution image tensor [B, C, H, W]
        hr_img (torch.Tensor): High-resolution image tensor [B, C, H, W]
        epoch (int): Current epoch
        batch_idx (int): Batch index
        save_path (str): Path to save visualization
    """
    # Convert tensors to numpy arrays
    if isinstance(lr_img, torch.Tensor):
        lr_img = lr_img.detach().cpu().numpy()
    if isinstance(sr_img, torch.Tensor):
        sr_img = sr_img.detach().cpu().numpy()
    if isinstance(hr_img, torch.Tensor):
        hr_img = hr_img.detach().cpu().numpy()
    
    # Ensure images are in range [0, 1]
    lr_img = np.clip(lr_img, 0, 1)
    sr_img = np.clip(sr_img, 0, 1)
    hr_img = np.clip(hr_img, 0, 1)
    
    # Get number of samples to visualize
    n_samples = min(lr_img.shape[0], 4)  # Show at most 4 samples
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Low-resolution image
        axes[i, 0].imshow(lr_img[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Low-Resolution')
        axes[i, 0].axis('off')
        
        # Super-resolution image
        axes[i, 1].imshow(sr_img[i, 0], cmap='gray')
        axes[i, 1].set_title(f'Super-Resolution (PSNR: {calculate_psnr(sr_img[i, 0], hr_img[i, 0]):.2f}dB)')
        axes[i, 1].axis('off')
        
        # High-resolution image
        axes[i, 2].imshow(hr_img[i, 0], cmap='gray')
        axes[i, 2].set_title(f'High-Resolution (Ground Truth)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def AverageMeter():
    """
    Computes and stores the average and current value.
    """
    class _AverageMeter:
        def __init__(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    return _AverageMeter() 