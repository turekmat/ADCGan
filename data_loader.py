"""
Data loader for the Super-Resolution GAN model.
Handles loading 3D NIfTI and MHA files and converting them to 2D slices.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk
from torchvision import transforms
from torch.nn import functional as F
from config import DATA, AUGMENTATION
import matplotlib.pyplot as plt

class MedicalImageDataset(Dataset):
    """
    Dataset for medical image super-resolution.
    Loads 3D NIfTI (.nii) or MHA files and extracts 2D slices.
    """
    
    def __init__(self, data_dir=None, is_train=True, file_extension='.nii', file_list=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str, optional): Directory containing the medical image files.
                                     Not required if file_list is provided.
            is_train (bool): Whether this is for training or validation
            file_extension (str): File extension of the medical images ('.nii' or '.mha')
            file_list (list, optional): List of specific file paths to use.
                                       If provided, data_dir is ignored.
        """
        self.is_train = is_train
        self.file_extension = file_extension
        
        # Get the file list either from data_dir or directly from file_list parameter
        if file_list is not None:
            self.file_list = file_list
        elif data_dir is not None:
            self.file_list = [
                os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.endswith(file_extension)
            ]
        else:
            raise ValueError("Either data_dir or file_list must be provided")
        
        # Store 3D volume slices that have sufficient content (not all zeros)
        self.valid_slices = []
        self._preprocess_data()
        
        # Transformations for data augmentation
        if is_train:
            # RandomRotation doesn't support 'p' parameter, so we use RandomApply to apply it with a probability
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=AUGMENTATION['flip_probability']),
                transforms.RandomVerticalFlip(p=AUGMENTATION['flip_probability']),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=AUGMENTATION['max_rotation_angle'])],
                    p=AUGMENTATION['rotation_probability']
                )
            ])
        else:
            self.transform = transforms.Lambda(lambda x: x)  # Identity transform for validation
    
    def _preprocess_data(self):
        """
        Preprocess all 3D volumes and identify valid 2D slices.
        Stores tuples of (file_path, slice_idx) for valid slices.
        """
        print(f"Preprocessing data from {len(self.file_list)} files...")
        
        for file_path in self.file_list:
            # Load the 3D volume
            if self.file_extension == '.nii' or file_path.endswith('.nii'):
                volume = self._load_nifti(file_path)
            else:  # '.mha'
                volume = self._load_mha(file_path)
            
            # Find valid slices (not all zeros)
            for slice_idx in range(volume.shape[2]):
                slice_img = volume[:, :, slice_idx]
                
                # Check if slice has sufficient content
                if slice_img.max() > DATA['slice_threshold']:
                    self.valid_slices.append((file_path, slice_idx))
        
        print(f"Found {len(self.valid_slices)} valid 2D slices")
    
    def _load_nifti(self, file_path):
        """
        Load a NIfTI file and return the volume data.
        
        Args:
            file_path (str): Path to the NIfTI file
            
        Returns:
            numpy.ndarray: 3D volume data normalized to [0, 1]
        """
        nifti_img = nib.load(file_path)
        volume = nifti_img.get_fdata()
        
        # Assuming data is already normalized to [0, 1]
        # If not, uncomment the following line
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        return volume
    
    def _load_mha(self, file_path):
        """
        Load an MHA file and return the volume data.
        
        Args:
            file_path (str): Path to the MHA file
            
        Returns:
            numpy.ndarray: 3D volume data normalized to [0, 1]
        """
        itk_img = sitk.ReadImage(file_path)
        volume = sitk.GetArrayFromImage(itk_img)
        
        # Convert from [slices, height, width] to [height, width, slices]
        volume = np.transpose(volume, (1, 2, 0))
        
        # Assuming data is already normalized to [0, 1]
        # If not, uncomment the following line
        # volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        return volume
    
    def __len__(self):
        """
        Return the number of valid 2D slices.
        """
        return len(self.valid_slices)
    
    def __getitem__(self, idx):
        """
        Get a single 2D slice and create its downsampled version.
        
        Args:
            idx (int): Index of the slice
            
        Returns:
            tuple: (lr_image, hr_image) tensors
        """
        file_path, slice_idx = self.valid_slices[idx]
        
        # Load the 3D volume
        if self.file_extension == '.nii' or file_path.endswith('.nii'):
            volume = self._load_nifti(file_path)
        else:  # '.mha'
            volume = self._load_mha(file_path)
        
        # Extract the 2D slice
        hr_image = volume[:, :, slice_idx]
        
        # Convert to tensor with channel dimension [1, H, W]
        hr_image = torch.from_numpy(hr_image).float().unsqueeze(0)
        
        # Resize if needed to match the configured high-resolution size
        current_h, current_w = hr_image.shape[1], hr_image.shape[2]
        if current_h != DATA['hr_size'] or current_w != DATA['hr_size']:
            hr_image = F.interpolate(
                hr_image.unsqueeze(0),
                size=(DATA['hr_size'], DATA['hr_size']),
                mode='bicubic',
                align_corners=False
            ).squeeze(0)
        
        # Apply data augmentation if training
        if self.is_train:
            hr_image = self.transform(hr_image)
        
        # Create low-resolution image through downsampling
        lr_image = F.interpolate(
            hr_image.unsqueeze(0),
            scale_factor=1/DATA['scale_factor'],
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        return lr_image, hr_image


def get_dataloader(data_dir, batch_size, is_train=True, file_extension='.nii', num_workers=4):
    """
    Create a DataLoader for medical images.
    
    Args:
        data_dir (str): Directory containing the medical image files
        batch_size (int): Batch size
        is_train (bool): Whether this is for training or validation
        file_extension (str): File extension of the medical images ('.nii' or '.mha')
        num_workers (int): Number of worker threads for loading data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset
    """
    dataset = MedicalImageDataset(data_dir, is_train, file_extension)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
    )


# For testing
if __name__ == "__main__":
    # Test dataset on a sample directory
    test_dir = "/path/to/test/data"
    if os.path.exists(test_dir):
        dataset = MedicalImageDataset(test_dir)
        if len(dataset) > 0:
            # Get a sample and visualize it
            lr_img, hr_img = dataset[0]
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(lr_img.squeeze(0), cmap='gray')
            plt.title(f'Low-Resolution ({lr_img.shape[1]}x{lr_img.shape[2]})')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(hr_img.squeeze(0), cmap='gray')
            plt.title(f'High-Resolution ({hr_img.shape[1]}x{hr_img.shape[2]})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"Dataset size: {len(dataset)}")
            print(f"LR shape: {lr_img.shape}, HR shape: {hr_img.shape}")
        else:
            print("No valid slices found in the dataset")
    else:
        print(f"Test directory {test_dir} does not exist") 