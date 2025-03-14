"""
Configuration settings for the Super-Resolution GAN for medical imaging.
"""

# Data settings
DATA = {
    'hr_size': 256,  # High-resolution image size
    'lr_size': 64,  # Low-resolution image size (downsampled by 4x)
    'scale_factor': 4,  # Super-resolution scaling factor
    'slice_threshold': 0.01,  # Threshold to skip black slices (if max value < threshold)
    'channels': 1,  # Number of channels in the input images (1 for grayscale)
}

# Augmentation settings
AUGMENTATION = {
    'flip_probability': 0.5,  # Probability of horizontal/vertical flip
    'rotation_probability': 0.6,  # Probability of rotation
    'max_rotation_angle': 15,  # Maximum rotation angle in degrees
}

# Model architecture
MODEL = {
    'generator': {
           'n_residual_blocks': 23,  # Zvýšit z 16
           'base_filters': 96,       # Zvýšit z 64
       },
    'discriminator': {
        'base_filters': 64,  # Number of base filters in the discriminator
    },
}

# Training settings
TRAINING = {
    'batch_size': 16,
    'num_epochs': 100,
    'lr_generator': 0.00005,  # Learning rate for the generator
    'lr_discriminator': 0.0001,  # Learning rate for the discriminator
    'beta1': 0.9,  # Beta1 for Adam optimizer
    'beta2': 0.999,  # Beta2 for Adam optimizer
    'content_loss_weight': 1.0,  # Weight for content (L1) loss
    'adversarial_loss_weight': 0.0005,  # Weight for adversarial loss
    'perceptual_loss_weight': 0.01,  # Weight for perceptual loss
    'save_frequency': 1,  # Save model after every N epochs
    'log_frequency': 100,  # Log after every N batches
    'validation_frequency': 1,  # Validate after every N epochs
}

# Directories
DIRS = {
    'checkpoints': './checkpoints',
    'results': './results',
    'logs': './logs',
}

# Visualization
VISUALIZATION = {
    'max_samples': 4,  # Maximum number of samples to visualize
} 