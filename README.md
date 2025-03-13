# Medical Image Super-Resolution using GAN

A GAN-based super-resolution model specifically designed for medical imaging, especially for brain ADC maps.

## Features

- 2× super-resolution for 3D medical images
- Processes 3D volumes by slicing them into 2D images
- Supports both NIfTI (.nii) and MHA file formats
- Implements data augmentation (flips and rotations)
- Preserves information integrity critical for medical imaging
- Progressive model saving and visualization

## Requirements

- Python 3.7+
- PyTorch 1.9+
- See requirements.txt for all dependencies

## Project Structure

```
├── data_loader.py      # Data loading utilities for NIfTI and MHA files
├── models/
│   ├── generator.py    # Generator network architecture
│   ├── discriminator.py # Discriminator network architecture
│   └── srgan.py        # SRGAN model implementation
├── train.py            # Training script
├── inference.py        # Inference script for new data
├── utils.py            # Utility functions
└── config.py           # Configuration parameters
```

## Usage

### Training

```bash
python train.py --data_dir /path/to/nii/files --epochs 100 --batch_size 16
```

### Inference

```bash
python inference.py --input /path/to/input.mha --output /path/to/output.mha --checkpoint /path/to/model.pth
```

## Data Preprocessing

The model assumes input data that has been:

1. Normalized to range [0, 1]
2. Cropped to the region of interest (bounding box)

## Notes for Medical Imaging

This implementation prioritizes information preservation in medical images by avoiding unnecessary compression or transformation of image data beyond the required downsampling for super-resolution.
