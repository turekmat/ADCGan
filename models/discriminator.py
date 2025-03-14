"""
Discriminator model for the SRGAN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL, DATA

class DiscriminatorBlock(nn.Module):
    """
    Basic block for the discriminator.
    """
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super(DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.lrelu(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator network for SRGAN.
    
    Architecture:
        1. Initial convolutional layer
        2. Sequence of discriminator blocks with increasing channels and downsampling
        3. Flattening and fully connected layers for binary classification
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Parameters
        base_filters = MODEL['discriminator']['base_filters']
        
        # Initial convolution block without batch normalization
        self.initial = DiscriminatorBlock(DATA['channels'], base_filters, stride=1, batch_norm=False)
        
        # Sequence of discriminator blocks with strided convolutions
        self.blocks = nn.Sequential(
            DiscriminatorBlock(base_filters, base_filters, stride=2),
            DiscriminatorBlock(base_filters, base_filters * 2, stride=1),
            DiscriminatorBlock(base_filters * 2, base_filters * 2, stride=2),
            DiscriminatorBlock(base_filters * 2, base_filters * 4, stride=1),
            DiscriminatorBlock(base_filters * 4, base_filters * 4, stride=2),
            DiscriminatorBlock(base_filters * 4, base_filters * 8, stride=1),
            DiscriminatorBlock(base_filters * 8, base_filters * 8, stride=2),
        )
        
        # Dynamicky zvýšit kapacitu diskriminátoru pro 4x
        if DATA['scale_factor'] == 4:
            print(f"Using enhanced discriminator architecture for 4x upscaling")
            # Přidat další bloky pro zpracování většího rozlišení při 4x
            self.blocks.add_module('extra1', DiscriminatorBlock(base_filters * 8, base_filters * 16, stride=1))
            self.blocks.add_module('extra2', DiscriminatorBlock(base_filters * 16, base_filters * 16, stride=2))
            final_filters = base_filters * 16
        else:
            final_filters = base_filters * 8
        
        # Adaptive average pooling zajistí, že výstup bude vždy 1x1 bez ohledu na vstupní velikost
        # Toto dělá výpočet feature_size nepotřebným a model je adaptivní pro různé scale_factory
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_filters, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial convolution
        x = self.initial(x)
        
        # Discriminator blocks
        x = self.blocks(x)
        
        # Classification
        x = self.classifier(x)
        
        return x


# For testing
if __name__ == "__main__":
    # Create a discriminator model
    discriminator = Discriminator()
    
    # Print model summary
    print(discriminator)
    
    # Test with a sample input
    sample_input = torch.randn(1, DATA['channels'], DATA['hr_size'], DATA['hr_size'])
    sample_output = discriminator(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Total parameters: {total_params:,}") 