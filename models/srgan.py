"""
SRGAN model integrating generator and discriminator with loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from models.generator import Generator
from models.discriminator import Discriminator
from config import TRAINING

class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    """
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()
        # Load VGG19 model pretrained on ImageNet
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
        self.vgg = nn.Sequential()
        self.vgg.add_module('0', vgg[0])  # conv1_1
        
        # We use the output of the first convolutional layer
        # This is sufficient for our medical imaging task while being efficient
        
        # Freeze the VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.MSELoss()
        
    def forward(self, sr, hr):
        # For grayscale images, repeat to create 3 channels for VGG
        if sr.size(1) == 1:
            sr = sr.repeat(1, 3, 1, 1)
            hr = hr.repeat(1, 3, 1, 1)
        
        # Extract VGG features
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        
        # Calculate MSE loss between features
        return self.criterion(sr_features, hr_features)


class SRGAN(nn.Module):
    """
    Super-Resolution Generative Adversarial Network (SRGAN).
    Combines generator and discriminator with appropriate loss functions.
    """
    def __init__(self, device='cuda'):
        super(SRGAN, self).__init__()
        
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        # Define loss functions
        self.content_criterion = nn.L1Loss()  # L1 loss for pixel-wise content
        self.adversarial_criterion = nn.BCELoss()  # BCE loss for GAN
        self.perceptual_criterion = VGGLoss(device)  # Perceptual loss using VGG
        
        # Loss weights
        self.content_weight = TRAINING['content_loss_weight']
        self.adversarial_weight = TRAINING['adversarial_loss_weight']
        self.perceptual_weight = TRAINING['perceptual_loss_weight']
        
    def generator_loss(self, sr_img, hr_img, fake_preds):
        """
        Calculate generator loss.
        
        Args:
            sr_img (torch.Tensor): Super-resolution images
            hr_img (torch.Tensor): High-resolution images
            fake_preds (torch.Tensor): Discriminator predictions for SR images
            
        Returns:
            tuple: Total generator loss, content loss, adversarial loss, perceptual loss
        """
        # Content loss (L1)
        content_loss = self.content_criterion(sr_img, hr_img)
        
        # Adversarial loss (trying to fool discriminator)
        target_real = torch.ones_like(fake_preds).to(self.device)
        adversarial_loss = self.adversarial_criterion(fake_preds, target_real)
        
        # Perceptual loss (VGG features)
        perceptual_loss = self.perceptual_criterion(sr_img, hr_img)
        
        # Combine losses with weights
        total_loss = (
            self.content_weight * content_loss +
            self.adversarial_weight * adversarial_loss +
            self.perceptual_weight * perceptual_loss
        )
        
        return total_loss, content_loss, adversarial_loss, perceptual_loss
    
    def discriminator_loss(self, real_preds, fake_preds):
        """
        Calculate discriminator loss.
        
        Args:
            real_preds (torch.Tensor): Discriminator predictions for HR images
            fake_preds (torch.Tensor): Discriminator predictions for SR images
            
        Returns:
            torch.Tensor: Discriminator loss
        """
        # Real images should be classified as real (1)
        target_real = torch.ones_like(real_preds).to(self.device)
        loss_real = self.adversarial_criterion(real_preds, target_real)
        
        # Fake images should be classified as fake (0)
        target_fake = torch.zeros_like(fake_preds).to(self.device)
        loss_fake = self.adversarial_criterion(fake_preds, target_fake)
        
        # Total discriminator loss
        total_loss = (loss_real + loss_fake) / 2
        
        return total_loss


# For testing
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create SRGAN model
    srgan = SRGAN(device)
    
    # Test with sample data
    from config import DATA
    
    # Create sample data
    lr_img = torch.randn(2, DATA['channels'], DATA['lr_size'], DATA['lr_size']).to(device)
    hr_img = torch.randn(2, DATA['channels'], DATA['hr_size'], DATA['hr_size']).to(device)
    
    # Generator forward pass
    sr_img = srgan.generator(lr_img)
    
    # Discriminator forward passes
    real_preds = srgan.discriminator(hr_img)
    fake_preds = srgan.discriminator(sr_img)
    
    # Calculate losses
    g_loss, content_loss, adv_loss, perc_loss = srgan.generator_loss(sr_img, hr_img, fake_preds)
    d_loss = srgan.discriminator_loss(real_preds, fake_preds)
    
    print(f"Generator loss: {g_loss.item():.4f}")
    print(f"  - Content loss: {content_loss.item():.4f}")
    print(f"  - Adversarial loss: {adv_loss.item():.4f}")
    print(f"  - Perceptual loss: {perc_loss.item():.4f}")
    print(f"Discriminator loss: {d_loss.item():.4f}")
    
    # Check shapes
    print(f"LR shape: {lr_img.shape}")
    print(f"SR shape: {sr_img.shape}")
    print(f"HR shape: {hr_img.shape}")
    print(f"Disc predictions (real): {real_preds.shape}")
    print(f"Disc predictions (fake): {fake_preds.shape}") 