"""
Generator model for the SRGAN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MODEL, DATA

class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization.
    """
    def __init__(self, in_channels, out_channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        return out


class UpsampleBlock(nn.Module):
    """
    Upsampling block using pixel shuffle.
    """
    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """
    Generator network for SRGAN.
    
    Architecture:
        1. Initial convolutional layer
        2. Residual blocks with skip connections
        3. Upsampling blocks using pixel shuffle
        4. Final convolutional layer
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Parameters
        self.scale_factor = DATA['scale_factor']
        
        # Dynamicky upravit počet bloků a filtrů pro scale_factor=4
        if self.scale_factor == 4:
            # Pro 4x upscaling použijeme větší model
            n_residual_blocks = MODEL['generator']['n_residual_blocks'] + 8  # Zvýšit počet bloků o 8
            base_filters = MODEL['generator']['base_filters'] + 32  # Zvýšit počet filtrů o 32
            print(f"Using enhanced generator architecture for 4x upscaling: {n_residual_blocks} residual blocks, {base_filters} base filters")
        else:
            # Pro 2x upscaling použijeme standardní model
            n_residual_blocks = MODEL['generator']['n_residual_blocks']
            base_filters = MODEL['generator']['base_filters']
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(DATA['channels'], base_filters, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU()
        
        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(base_filters, base_filters))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Second convolutional layer after residual blocks
        self.conv2 = nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters)
        
        # Upsampling blocks
        upsample_blocks = []
        for _ in range(int(self.scale_factor / 2)):  # For 2x upscaling, we need 1 upsampling block
            upsample_blocks.append(UpsampleBlock(base_filters))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        # Final output convolutional layer
        self.conv3 = nn.Conv2d(base_filters, DATA['channels'], kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Initial convolutional layer
        x1 = self.prelu(self.conv1(x))
        
        # Residual blocks
        x = self.res_blocks(x1)
        
        # Convolutional layer after residual blocks with skip connection
        x = self.bn2(self.conv2(x))
        x = x + x1  # Global skip connection
        
        # Upsampling blocks
        x = self.upsample_blocks(x)
        
        # Final convolutional layer and activation
        x = self.tanh(self.conv3(x))
        
        # Scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        return x


# For testing
if __name__ == "__main__":
    # Create a generator model
    generator = Generator()
    
    # Print model summary
    print(generator)
    
    # Test with a sample input
    sample_input = torch.randn(1, DATA['channels'], DATA['lr_size'], DATA['lr_size'])
    sample_output = generator(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Total parameters: {total_params:,}") 