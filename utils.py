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

def calculate_psnr(img1, img2, mask=None):
    """
    Calculate PSNR between two images, optionally only in masked regions.
    
    Args:
        img1 (numpy.ndarray or torch.Tensor): First image
        img2 (numpy.ndarray or torch.Tensor): Second image
        mask (numpy.ndarray or torch.Tensor, optional): If provided, only calculate metrics in non-zero mask regions
        
    Returns:
        float: PSNR value
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create mask from img2 (HR image) if not provided
    if mask is None:
        # If img2 is 4D [B,C,H,W], create mask where any channel is non-zero
        if img2.ndim == 4:
            mask = np.any(img2 > 0, axis=1)
        else:
            mask = img2 > 0
    
    # Ensure images are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # For multi-batch images
    if img1.ndim == 4:  # [B, C, H, W]
        psnr_values = []
        for i in range(img1.shape[0]):
            # Extract single image
            single_img1 = img1[i, 0]  # Assuming single channel
            single_img2 = img2[i, 0]
            
            # Get appropriate mask for this batch item
            if mask.ndim == 4:  # Mask is [B, C, H, W]
                single_mask = mask[i, 0]
            elif mask.ndim == 3:  # Mask is [B, H, W]
                single_mask = mask[i]
            else:  # Mask is [H, W]
                single_mask = mask
            
            # Skip if mask is empty
            if np.sum(single_mask) == 0:
                continue
                
            # Apply mask - extract only the pixels where mask is True
            masked_img1 = single_img1[single_mask]
            masked_img2 = single_img2[single_mask]
            
            # Calculate PSNR on masked areas
            mse = np.mean((masked_img1 - masked_img2) ** 2)
            if mse == 0:
                psnr_values.append(100)  # To handle perfect reconstruction
            else:
                psnr_values.append(10 * np.log10(1.0 / mse))
        
        return np.mean(psnr_values) if psnr_values else 0.0
    else:
        # For single images (non-batched)
        
        # Extract the right dimensions for single-channel images
        if img1.ndim == 3:  # [C, H, W]
            # Use the first channel for simplicity
            img1_2d = img1[0]
            img2_2d = img2[0]
            
            # Get 2D mask from the first channel if necessary
            if mask.ndim == 3:
                mask_2d = mask[0]
            elif mask.ndim > 3:
                mask_2d = mask[0, 0]
            else:
                mask_2d = mask
        else:  # [H, W]
            img1_2d = img1
            img2_2d = img2
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask_2d = mask[0] if mask.ndim == 3 else mask[0, 0]
            else:
                mask_2d = mask
        
        # Skip if mask is empty
        if np.sum(mask_2d) == 0:
            return 0.0
        
        # Apply mask to get the masked pixels
        masked_img1 = img1_2d[mask_2d]
        masked_img2 = img2_2d[mask_2d]
        
        # Calculate PSNR on masked areas
        mse = np.mean((masked_img1 - masked_img2) ** 2)
        
        if mse == 0:
            return 100  # To handle perfect reconstruction
        return 10 * np.log10(1.0 / mse)

def calculate_ssim(img1, img2, mask=None):
    """
    Calculate Structural Similarity Index (SSIM) between two images, optionally only in masked regions.
    
    Args:
        img1 (numpy.ndarray or torch.Tensor): First image
        img2 (numpy.ndarray or torch.Tensor): Second image
        mask (numpy.ndarray or torch.Tensor, optional): If provided, only calculate metrics in non-zero mask regions
        
    Returns:
        float: SSIM value
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create mask from img2 (HR image) if not provided
    if mask is None:
        # If img2 is 4D [B,C,H,W], create mask where any channel is non-zero
        if img2.ndim == 4:
            mask = np.any(img2 > 0, axis=1)
        else:
            mask = img2 > 0
    
    # Ensure images are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # For multi-batch images
    if img1.ndim == 4:  # [B, C, H, W]
        ssim_values = []
        for i in range(img1.shape[0]):
            # Extract single image (first channel for simplicity)
            single_img1 = img1[i, 0]
            single_img2 = img2[i, 0]
            
            # Get appropriate mask for this batch item
            if mask.ndim == 4:  # Mask is [B, C, H, W]
                single_mask = mask[i, 0]
            elif mask.ndim == 3:  # Mask is [B, H, W]
                single_mask = mask[i]
            else:  # Mask is [H, W]
                single_mask = mask
            
            # Skip if mask is empty or too small
            if np.sum(single_mask) < 16:  # Minimum size needed for SSIM
                continue
                
            # For SSIM, we need to use the original shape but apply the mask
            # Create masked versions where non-mask areas are set to 0
            masked_img1 = single_img1.copy()
            masked_img2 = single_img2.copy()
            masked_img1[~single_mask] = 0
            masked_img2[~single_mask] = 0
            
            try:
                # Try computing SSIM on masked images
                ssim_val = structural_similarity(
                    masked_img1, masked_img2, data_range=1.0, multichannel=False
                )
                ssim_values.append(ssim_val)
            except Exception as e:
                print(f"Error computing SSIM: {e}")
                continue
        
        return np.mean(ssim_values) if ssim_values else 0.0
    else:
        # For single images (non-batched)
        
        # Extract the right dimensions for single-channel images
        if img1.ndim == 3:  # [C, H, W]
            # Use the first channel for simplicity
            img1_2d = img1[0]
            img2_2d = img2[0]
            
            # Get 2D mask from the first channel if necessary
            if mask.ndim == 3:
                mask_2d = mask[0]
            elif mask.ndim > 3:
                mask_2d = mask[0, 0]
            else:
                mask_2d = mask
        else:  # [H, W]
            img1_2d = img1
            img2_2d = img2
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask_2d = mask[0] if mask.ndim == 3 else mask[0, 0]
            else:
                mask_2d = mask
        
        # Skip if mask is empty or too small
        if np.sum(mask_2d) < 16:  # Minimum size needed for SSIM
            return 0.0
            
        # For SSIM, we need to use the original shape but apply the mask
        # Create masked versions where non-mask areas are set to 0
        masked_img1 = img1_2d.copy()
        masked_img2 = img2_2d.copy()
        masked_img1[~mask_2d] = 0
        masked_img2[~mask_2d] = 0
        
        try:
            # Try computing SSIM on masked images
            return structural_similarity(
                masked_img1, masked_img2, data_range=1.0, multichannel=False
            )
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            return 0.0

def calculate_mae(img1, img2, mask=None):
    """
    Calculate Mean Absolute Error (MAE) between two images, optionally only in masked regions.
    
    Args:
        img1 (numpy.ndarray or torch.Tensor): First image
        img2 (numpy.ndarray or torch.Tensor): Second image
        mask (numpy.ndarray or torch.Tensor, optional): If provided, only calculate metrics in non-zero mask regions
        
    Returns:
        float: MAE value
    """
    # Convert to numpy arrays if they are torch tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create mask from img2 (HR image) if not provided
    if mask is None:
        # If img2 is 4D [B,C,H,W], create mask where any channel is non-zero
        if img2.ndim == 4:
            mask = np.any(img2 > 0, axis=1)
        else:
            mask = img2 > 0
    
    # Ensure images are in range [0, 1]
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # For multi-batch images
    if img1.ndim == 4:  # [B, C, H, W]
        mae_values = []
        for i in range(img1.shape[0]):
            # Extract single image (first channel for simplicity)
            single_img1 = img1[i, 0]
            single_img2 = img2[i, 0]
            
            # Get appropriate mask for this batch item
            if mask.ndim == 4:  # Mask is [B, C, H, W]
                single_mask = mask[i, 0]
            elif mask.ndim == 3:  # Mask is [B, H, W]
                single_mask = mask[i]
            else:  # Mask is [H, W]
                single_mask = mask
            
            # Skip if mask is empty
            if np.sum(single_mask) == 0:
                continue
                
            # Apply mask and calculate MAE
            masked_diff = np.abs(single_img1 - single_img2) * single_mask
            mae_values.append(np.sum(masked_diff) / np.sum(single_mask))
        
        return np.mean(mae_values) if mae_values else 0.0
    else:
        # For single images (non-batched)
        
        # Extract the right dimensions for single-channel images
        if img1.ndim == 3:  # [C, H, W]
            # Use the first channel for simplicity
            img1_2d = img1[0]
            img2_2d = img2[0]
            
            # Get 2D mask from the first channel if necessary
            if mask.ndim == 3:
                mask_2d = mask[0]
            elif mask.ndim > 3:
                mask_2d = mask[0, 0]
            else:
                mask_2d = mask
        else:  # [H, W]
            img1_2d = img1
            img2_2d = img2
            
            # Ensure mask is 2D
            if mask.ndim > 2:
                mask_2d = mask[0] if mask.ndim == 3 else mask[0, 0]
            else:
                mask_2d = mask
        
        # Skip if mask is empty
        if np.sum(mask_2d) == 0:
            return 0.0
        
        # Apply mask and calculate MAE
        masked_diff = np.abs(img1_2d - img2_2d) * mask_2d
        return np.sum(masked_diff) / np.sum(mask_2d)

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

def visualize_results(lr_img, sr_img, hr_img, epoch, batch_idx=0, save_path=None, custom_title=None):
    """
    Visualize low-resolution, super-resolution, and high-resolution images.
    
    Args:
        lr_img (torch.Tensor): Low-resolution image tensor [B, C, H, W]
        sr_img (torch.Tensor): Super-resolution image tensor [B, C, H, W]
        hr_img (torch.Tensor): High-resolution image tensor [B, C, H, W]
        epoch (int): Current epoch
        batch_idx (int): Batch index
        save_path (str): Path to save visualization
        custom_title (str): Optional custom title for the figure
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
    
    # Determine if this is for PDF output
    is_pdf = save_path and save_path.endswith('.pdf')
    if is_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
    
    # Create figure with standard dimensions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract the middle slice for 3D volumes or use the entire 2D image
    lr_slice = lr_img[0, 0]
    sr_slice = sr_img[0, 0]
    hr_slice = hr_img[0, 0]
    
    # If the images are 3D (with multiple slices)
    if len(lr_slice.shape) > 2:  # If slices dimension exists
        middle_idx = lr_slice.shape[0] // 2  # Get middle slice index
        lr_slice = lr_slice[middle_idx]
        sr_slice = sr_slice[middle_idx]
        hr_slice = hr_slice[middle_idx]
    
    # Create mask from HR image
    mask = hr_slice > 0
    
    # Calculate metrics in masked region
    ssim_val = calculate_ssim(sr_slice, hr_slice, mask)
    mae_val = calculate_mae(sr_slice, hr_slice, mask)
    
    # Low-resolution image
    axes[0].imshow(lr_slice, cmap='gray')
    axes[0].set_title("Low-Resolution")
    axes[0].axis("off")
    
    # Super-resolution image
    axes[1].imshow(sr_slice, cmap='gray')
    axes[1].set_title(f"Our SRGAN 4x\nSSIM: {ssim_val:.4f}\nMAE: {mae_val:.4f}")
    axes[1].axis("off")
    
    # High-resolution image
    axes[2].imshow(hr_slice, cmap='gray')
    axes[2].set_title("High-Resolution\n(Ground Truth)")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # If PDF format is requested
        if is_pdf:
            with PdfPages(save_path) as pdf:
                pdf.savefig(fig)
            print(f"PDF comparison saved to: {save_path}")
        else:
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
