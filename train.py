"""
Training script for the SRGAN model.
"""

import os
import argparse
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.srgan import SRGAN
from data_loader import get_dataloader, MedicalImageDataset
from utils import create_dirs, visualize_results, AverageMeter, calculate_psnr, calculate_ssim
from config import TRAINING, DIRS, DATA, VISUALIZATION

def train_val_split(file_list, val_split=0.1, seed=42):
    """
    Split a list of files into training and validation sets.
    
    Args:
        file_list (list): List of file paths
        val_split (float): Fraction of files to use for validation
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (training_files, validation_files)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the file list
    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)
    
    # Calculate the split index
    val_size = max(1, int(len(shuffled_files) * val_split))
    
    # Split the files
    val_files = shuffled_files[:val_size]
    train_files = shuffled_files[val_size:]
    
    return train_files, val_files

def train(args):
    """
    Train the SRGAN model.
    
    Args:
        args: Command-line arguments
    """
    # Create necessary directories
    create_dirs()
    
    # Vypsat informace o konfiguraci tréninku
    print(f"Training SRGAN with {DATA['scale_factor']}x upscaling")
    print(f"HR size: {DATA['hr_size']}x{DATA['hr_size']}, LR size: {DATA['lr_size']}x{DATA['lr_size']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SRGAN(device)
    
    # Create optimizers
    optimizer_g = optim.Adam(
        model.generator.parameters(),
        lr=TRAINING['lr_generator'],
        betas=(TRAINING['beta1'], TRAINING['beta2'])
    )
    optimizer_d = optim.Adam(
        model.discriminator.parameters(),
        lr=TRAINING['lr_discriminator'],
        betas=(TRAINING['beta1'], TRAINING['beta2'])
    )
    
    # Handle data loading
    if args.val_dir:
        # If separate validation directory is provided, use it
        train_dataloader = get_dataloader(
            args.data_dir,
            batch_size=args.batch_size,
            is_train=True,
            file_extension=args.file_extension,
            num_workers=args.num_workers
        )
        
        val_dataloader = get_dataloader(
            args.val_dir,
            batch_size=args.batch_size,
            is_train=False,
            file_extension=args.file_extension,
            num_workers=args.num_workers
        )
    else:
        # If no validation directory, split the main directory
        print(f"Splitting data with validation ratio: {args.validation_split}")
        
        # Get all files with the given extension
        data_dir = args.data_dir
        file_list = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith(args.file_extension)
        ]
        print(f"Found {len(file_list)} files in {data_dir}")
        
        # Split into training and validation sets
        train_files, val_files = train_val_split(file_list, args.validation_split, args.seed)
        print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")
        
        # Create custom datasets with specific file lists
        train_dataset = MedicalImageDataset(
            data_dir=None,  # Not used when file_list is provided
            is_train=True,
            file_extension=args.file_extension,
            file_list=train_files
        )
        
        val_dataset = MedicalImageDataset(
            data_dir=None,  # Not used when file_list is provided
            is_train=False,
            file_extension=args.file_extension,
            file_list=val_files
        )
        
        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=DIRS['logs'])
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.generator.load_state_dict(checkpoint['generator'])
            model.discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Store sample images for visualization
    sample_lr = None
    sample_hr = None
    
    # Training loop
    total_steps = len(train_dataloader)
    print(f"Starting training from epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, args.epochs):
        model.generator.train()
        model.discriminator.train()
        
        # Initialize average meters
        g_losses = AverageMeter()
        content_losses = AverageMeter()
        adv_losses = AverageMeter()
        perceptual_losses = AverageMeter()
        d_losses = AverageMeter()
        psnr_metrics = AverageMeter()
        ssim_metrics = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for i, (lr_imgs, hr_imgs) in pbar:
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Move data to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Store sample images for visualization
            if sample_lr is None and sample_hr is None:
                sample_lr = lr_imgs[:VISUALIZATION['max_samples']].clone()
                sample_hr = hr_imgs[:VISUALIZATION['max_samples']].clone()
            
            # Generate super-resolution images
            sr_imgs = model.generator(lr_imgs)
            
            #------------------------
            # Train Discriminator
            #------------------------
            optimizer_d.zero_grad()
            
            # Predictions
            real_preds = model.discriminator(hr_imgs)
            fake_preds = model.discriminator(sr_imgs.detach())
            
            # Calculate discriminator loss
            d_loss = model.discriminator_loss(real_preds, fake_preds)
            
            # Backpropagate and update discriminator
            d_loss.backward()
            optimizer_d.step()
            
            #------------------------
            # Train Generator
            #------------------------
            optimizer_g.zero_grad()
            
            # Recalculate predictions for generator update
            fake_preds = model.discriminator(sr_imgs)
            
            # Calculate generator loss
            g_loss, content_loss, adv_loss, perceptual_loss = model.generator_loss(
                sr_imgs, hr_imgs, fake_preds
            )
            
            # Backpropagate and update generator
            g_loss.backward()
            optimizer_g.step()
            
            # Calculate image quality metrics
            with torch.no_grad():
                psnr = calculate_psnr(sr_imgs.detach(), hr_imgs.detach())
                ssim = calculate_ssim(sr_imgs.detach(), hr_imgs.detach())
            
            # Update statistics
            g_losses.update(g_loss.item(), lr_imgs.size(0))
            content_losses.update(content_loss.item(), lr_imgs.size(0))
            adv_losses.update(adv_loss.item(), lr_imgs.size(0))
            perceptual_losses.update(perceptual_loss.item(), lr_imgs.size(0))
            d_losses.update(d_loss.item(), lr_imgs.size(0))
            psnr_metrics.update(psnr, lr_imgs.size(0))
            ssim_metrics.update(ssim, lr_imgs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update progress bar
            pbar.set_description(
                f"Epoch: [{epoch + 1}/{args.epochs}] "
                f"Batch: [{i + 1}/{total_steps}] "
                f"Time: {batch_time.avg:.4f} "
                f"G Loss: {g_losses.avg:.4f} "
                f"D Loss: {d_losses.avg:.4f} "
                f"PSNR: {psnr_metrics.avg:.2f}"
            )
            
            # Log to tensorboard
            current_step = epoch * len(train_dataloader) + i
            if (i + 1) % TRAINING['log_frequency'] == 0:
                writer.add_scalar('Train/G_Loss', g_losses.avg, current_step)
                writer.add_scalar('Train/Content_Loss', content_losses.avg, current_step)
                writer.add_scalar('Train/Adversarial_Loss', adv_losses.avg, current_step)
                writer.add_scalar('Train/Perceptual_Loss', perceptual_losses.avg, current_step)
                writer.add_scalar('Train/D_Loss', d_losses.avg, current_step)
                writer.add_scalar('Train/PSNR', psnr_metrics.avg, current_step)
                writer.add_scalar('Train/SSIM', ssim_metrics.avg, current_step)
        
        # Validate after each epoch if validation set is provided
        if val_dataloader is not None and (epoch + 1) % TRAINING['validation_frequency'] == 0:
            validate(val_dataloader, model, epoch, writer, device)
        
        # Save model checkpoint after each epoch
        if (epoch + 1) % TRAINING['save_frequency'] == 0:
            checkpoint_path = os.path.join(DIRS['checkpoints'], f'srgan_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'generator': model.generator.state_dict(),
                'discriminator': model.discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Visualize results after each epoch
        with torch.no_grad():
            model.generator.eval()
            sr_samples = model.generator(sample_lr)
            model.generator.train()
            
            # Save visualization
            vis_path = os.path.join(DIRS['results'], f'epoch_{epoch + 1}_results.png')
            visualize_results(sample_lr.cpu(), sr_samples.cpu(), sample_hr.cpu(), epoch + 1, save_path=vis_path)
            print(f"Visualization saved to {vis_path}")
    
    # Save final model
    final_model_path = os.path.join(DIRS['checkpoints'], 'srgan_final.pth')
    torch.save({
        'generator': model.generator.state_dict(),
        'discriminator': model.discriminator.state_dict(),
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    writer.close()


def validate(val_dataloader, model, epoch, writer, device):
    """
    Validate the model on the validation set.
    
    Args:
        val_dataloader: Validation dataloader
        model: SRGAN model
        epoch: Current epoch
        writer: Tensorboard writer
        device: Device to use
    """
    model.generator.eval()
    model.discriminator.eval()
    
    # Initialize average meters
    psnr_metrics = AverageMeter()
    ssim_metrics = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        for i, (lr_imgs, hr_imgs) in pbar:
            # Move data to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate super-resolution images
            sr_imgs = model.generator(lr_imgs)
            
            # Calculate image quality metrics
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            ssim = calculate_ssim(sr_imgs, hr_imgs)
            
            # Update statistics
            psnr_metrics.update(psnr, lr_imgs.size(0))
            ssim_metrics.update(ssim, lr_imgs.size(0))
            
            pbar.set_description(
                f"Validation: "
                f"PSNR: {psnr_metrics.avg:.2f} "
                f"SSIM: {ssim_metrics.avg:.4f}"
            )
    
    print(f"Validation Results - Epoch: {epoch + 1} "
          f"PSNR: {psnr_metrics.avg:.2f} "
          f"SSIM: {ssim_metrics.avg:.4f}")
    
    # Log to tensorboard
    writer.add_scalar('Val/PSNR', psnr_metrics.avg, epoch + 1)
    writer.add_scalar('Val/SSIM', ssim_metrics.avg, epoch + 1)
    
    # Restore model to training mode
    model.generator.train()
    model.discriminator.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SRGAN for medical image super-resolution')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with medical image files')
    parser.add_argument('--val-dir', type=str, default=None, help='Optional separate directory with validation files')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--file-extension', type=str, default='.nii', choices=['.nii', '.mha'], help='File extension')
    parser.add_argument('--epochs', type=int, default=TRAINING['num_epochs'], help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=TRAINING['batch_size'], help='Batch size')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--scale-factor', type=int, default=DATA['scale_factor'], choices=[2, 4], 
                      help='Super-resolution scaling factor (2x or 4x)')
    
    args = parser.parse_args()
    
    # Dynamicky nastavit scale_factor, pokud se liší od výchozí hodnoty
    if args.scale_factor != DATA['scale_factor']:
        print(f"Changing scale factor from {DATA['scale_factor']}x to {args.scale_factor}x")
        DATA['scale_factor'] = args.scale_factor
        # Upravit také lr_size podle scale_factoru
        DATA['lr_size'] = DATA['hr_size'] // args.scale_factor
        print(f"Adjusted lr_size to {DATA['lr_size']}")
    
    train(args) 