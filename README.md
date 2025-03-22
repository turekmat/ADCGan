# ADCGan: Advanced Medical Image Super-Resolution using GANs

A Generative Adversarial Network-based super-resolution model specifically designed for enhancing the quality of medical images, with special focus on diffusion MRI, particularly ADC (Apparent Diffusion Coefficient) and ZADC (z-scored ADC) maps.

## Overview

This project implements a modified SRGAN (Super-Resolution Generative Adversarial Network) architecture optimized for medical imaging applications. The model can upscale low-resolution medical images with either 2× or 4× scaling factors while preserving important diagnostic details.

## Key Features

- **Specialized for Medical Imaging**: Optimized for diffusion MRI data (ADC and ZADC maps)
- **Flexible Scaling Factors**: Support for both 2× and 4× super-resolution
- **Multiple File Formats**: Processes NIfTI (.nii) and MHA file formats
- **3D Volume Processing**: Handles 3D volumes by processing 2D slices
- **Adaptive Normalization**: Supports both standard ADC maps (0-1 range) and ZADC maps (-10 to 10 range)
- **Data Augmentation**: Implements rotation and flipping to improve generalization
- **Progressive Training**: Provides checkpoints, visualizations, and metrics during training
- **Masked Evaluation**: Calculates metrics (PSNR, SSIM, MAE) only on relevant image regions
- **TensorBoard Integration**: Logs progress for real-time monitoring

## Technical Details

The model uses a modified SRGAN architecture:

- Generator: ResNet-based architecture with pixel-shuffle upsampling
- Discriminator: Convolutional network that classifies images as real or generated
- Loss function: Combination of content loss (L1), adversarial loss, and perceptual loss
- Dynamic architecture sizing based on the chosen scaling factor

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/ADCGan.git
cd ADCGan
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data-dir /path/to/images --scale-factor 2 --data-range adc
```

#### Training Parameters

| Parameter            | Description                                      | Default    | Options           |
| -------------------- | ------------------------------------------------ | ---------- | ----------------- |
| `--data-dir`         | Directory containing training image files        | (Required) | Path to directory |
| `--val-dir`          | Optional separate directory for validation files | None       | Path to directory |
| `--validation-split` | Fraction of data to use for validation           | 0.1        | Float (0.0-1.0)   |
| `--file-extension`   | File extension of medical images                 | '.nii'     | '.nii', '.mha'    |
| `--epochs`           | Number of training epochs                        | 100        | Integer           |
| `--batch-size`       | Batch size for training                          | 16         | Integer           |
| `--resume`           | Path to checkpoint to resume training from       | None       | Path to .pth file |
| `--num-workers`      | Number of data loading workers                   | 4          | Integer           |
| `--no-cuda`          | Disable CUDA (use CPU)                           | False      | Flag              |
| `--seed`             | Random seed for reproducibility                  | 42         | Integer           |
| `--scale-factor`     | Super-resolution scaling factor                  | 4          | 2, 4              |
| `--data-range`       | Data range type                                  | 'adc'      | 'adc', 'zadc'     |

### Inference

```bash
python inference.py --input /path/to/input.nii --output /path/to/output.nii --checkpoint /path/to/model.pth --data-range zadc
```

#### Inference Parameters

| Parameter          | Description                                       | Default    | Options           |
| ------------------ | ------------------------------------------------- | ---------- | ----------------- |
| `--input`          | Input medical image file                          | (Required) | Path to file      |
| `--output`         | Output medical image file                         | (Required) | Path to file      |
| `--checkpoint`     | Path to model checkpoint                          | (Required) | Path to .pth file |
| `--file-extension` | File extension of input/output files              | '.nii'     | '.nii', '.mha'    |
| `--ground-truth`   | Ground truth high-resolution image for evaluation | None       | Path to file      |
| `--visualize`      | Create visualization of results                   | False      | Flag              |
| `--no-cuda`        | Disable CUDA (use CPU)                            | False      | Flag              |
| `--scale-factor`   | Super-resolution scaling factor                   | 4          | 2, 4              |
| `--data-range`     | Data range type                                   | 'adc'      | 'adc', 'zadc'     |

## Data Requirements

### Supported Data Types

1. **ADC Maps** (`--data-range adc`):

   - Values typically in range [0, 1]
   - If outside this range, min-max normalization is applied automatically

2. **ZADC Maps** (`--data-range zadc`):
   - Values typically in range [-10, 10]
   - Standardized normalization preserves the full range of values

### Data Preparation

For optimal results:

- Ensure consistent spatial resolution across your dataset
- The model works best when images have reasonable contrast
- Non-relevant areas (background) should ideally be zero-valued
- For 3D volumes, the model processes each slice independently

## Project Structure

```
├── config.py           # Configuration parameters
├── data_loader.py      # Data loading utilities
├── inference.py        # Inference script
├── models/
│   ├── discriminator.py # Discriminator network
│   ├── generator.py    # Generator network
│   └── srgan.py        # SRGAN model implementation
├── train.py            # Training script
└── utils.py            # Utility functions
```

## Output and Monitoring

- **Checkpoints**: Saved in `./checkpoints/` directory
- **Visualizations**: Generated in `./results/` directory
- **Logs**: TensorBoard logs in `./logs/` directory

To view training progress:

```bash
tensorboard --logdir=./logs
```

## Notes on Model Selection

- **For 2× Super-Resolution**: More stable, produces sharper results with fewer artifacts
- **For 4× Super-Resolution**: More computationally intensive, might require more training data for stable results
- For diffusion MRI data with negative values (like ZADC maps), ensure you use `--data-range zadc`

## Citation

If you use this code for your research, cite:

```
@misc{ADCGan2023,
  author = {Matyáš Turek},
  title = {ADCGan: Advanced Medical Image Super-Resolution using GANs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/turekmat/ADCGan}}
}
```
