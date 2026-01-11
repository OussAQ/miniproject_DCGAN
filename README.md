# Insect Image Generation using DCGAN

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic insect images. The implementation includes standard DCGAN architecture with some enhancements for better image quality.

## Overview

The model is trained on a dataset of insect images organized by categories (Ant, Bee, Beetle, Butterfly, Dragonfly, Fly, Grasshopper, Ladybug, Mosquito, Spider, Wasp). The DCGAN architecture uses standard techniques with some improvements for sharper image generation.

## Key Features

### Architecture Enhancements
- **Enhanced Generator**: Increased capacity (256 filters) for better detail generation
- **Texture-Sensitive Discriminator**: Uses minibatch standard deviation layer to encourage high-frequency detail
- **Label Smoothing**: Reduces overconfidence and improves generalization
- **Standard DCGAN Architecture**: Proven stable architecture for image generation

### Training Features
- **Binary Cross-Entropy (BCE) Loss**: Standard GAN loss with label smoothing
- **Equal Learning Rates**: Both generator and discriminator use the same learning rate (0.0002)
- **1:1 Training Balance**: One discriminator update per generator update
- **Adam Optimizer**: Standard DCGAN optimizer settings (beta1=0.5, beta2=0.999)

## Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/OussAQ/miniproject_DCGAN.git
cd miniproject_DCGAN
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset:**
   - Organize insect images in subdirectories by category
   - Place all images in a `data/` directory
   - Supported formats: `.jpg`, `.jpeg`, `.png`

## Usage

### Running the Notebook

1. **Start Jupyter Notebook:**
```bash
jupyter notebook insect_gan_training.ipynb
```

2. **Run cells sequentially:**
   - The notebook is organized into logical sections
   - Run all cells from top to bottom for complete training
   - Each section includes detailed explanations

### Notebook Structure

The notebook is organized into the following sections:

1. **Imports and Setup**: Library imports and hyperparameter configuration
2. **Dataset Class**: Custom dataset loader for insect images
3. **Generator Network**: DCGAN generator architecture with increased capacity
4. **Discriminator Network**: Texture-sensitive discriminator with minibatch std
5. **Optimizers**: Adam optimizers with equal learning rates
6. **Training Loop**: Main training procedure with loss tracking
7. **Visualization**: Loss plots and sample generation
8. **Model Utilities**: Checkpoint loading and custom sample generation

For detailed documentation of each cell, see [NOTEBOOK_DOCUMENTATION.md](NOTEBOOK_DOCUMENTATION.md).

## Architecture Details

### Generator Architecture

- **Input**: 100-dimensional noise vector
- **Architecture**: 
  - Fully connected layer: 100 → 4×4×2048
  - Transposed convolutions: 4×4 → 8×8 → 16×16 → 32×32 → 64×64
  - Batch normalization (except final layer)
  - ReLU activations (Tanh for output)
- **Output**: 64×64×3 RGB images
- **Capacity**: 256 filters (increased from standard 128 for better detail)

### Discriminator Architecture

- **Input**: 64×64×3 RGB images
- **Architecture**:
  - 4 convolutional layers with increasing depth
  - Strided convolutions: 64×64 → 32×32 → 16×16 → 8×8 → 4×4
  - Batch normalization (except first and last layers)
  - LeakyReLU activations (slope=0.2)
  - **Minibatch Standard Deviation**: Added before final layer to encourage texture detail
- **Output**: Single probability (real/fake) via sigmoid

### Key Design Choices

1. **No BatchNorm in Generator's Final Layer**: Important for maintaining sharpness
2. **No BatchNorm in Discriminator's First Layer**: Standard DCGAN practice
3. **No BatchNorm in Discriminator's Last Layer**: Before minibatch std layer
4. **Minibatch Standard Deviation**: Forces generator to produce high-frequency detail

## Hyperparameters

### Core Parameters
- **Image Size**: 64×64 pixels
- **Batch Size**: 64
- **Noise Dimension**: 100
- **Learning Rate**: 0.0002 (same for both generator and discriminator)
- **Epochs**: 100
- **Adam Betas**: (0.5, 0.999)

### Loss Configuration
- **Label Smoothing**: 0.1 (real labels: 0.9, fake labels: 0.1)
- **Loss Function**: Binary Cross-Entropy (BCE)

### Architecture
- **Generator Filters**: 256 (increased capacity)
- **Discriminator Filters**: 128 (base)
- **Training Balance**: 1:1 (one D update per G update)

## Training Process

### Loss Functions

**Discriminator Loss:**
$$L_D = -\mathbb{E}[\log D(x_{real})] - \mathbb{E}[\log(1 - D(x_{fake}))]$$

With label smoothing:
- Real labels: $1 - \alpha = 0.9$
- Fake labels: $\alpha = 0.1$

**Generator Loss:**
$$L_G = -\mathbb{E}[\log D(x_{fake})]$$

The generator tries to maximize the discriminator's probability on fake images.

### Training Steps

1. **Discriminator Training**:
   - Sample real images from dataset
   - Generate fake images from noise
   - Compute BCE loss with label smoothing
   - Update discriminator

2. **Generator Training**:
   - Generate fake images from noise
   - Compute adversarial loss (BCE)
   - Update generator

3. **Monitoring**:
   - Track losses for both networks
   - Generate sample images periodically
   - Save checkpoints every 10 epochs

## Project Structure

```
miniproject_DCGAN/
├── data/                          # Insect image dataset (organized by category)
│   ├── Ant/
│   ├── Bee/
│   ├── Beetle/
│   └── ...
├── checkpoints/                   # Saved model checkpoints
│   └── checkpoint_epoch_*.pth
├── samples/                       # Generated sample images
│   └── epoch_*.png
├── insect_gan_training.ipynb      # Main training notebook
├── NOTEBOOK_DOCUMENTATION.md      # Detailed notebook documentation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                    # Git ignore rules
```

## Model Checkpoints

- Checkpoints are saved every 10 epochs
- Each checkpoint includes:
  - Generator and discriminator state dictionaries
  - Optimizer states
  - Training losses history
  - Epoch number

**Loading a checkpoint:**
```python
checkpoint = torch.load('checkpoints/checkpoint_epoch_50.pth')
netG.load_state_dict(checkpoint['generator_state_dict'])
netD.load_state_dict(checkpoint['discriminator_state_dict'])
```

## Generated Samples

Sample images are automatically generated and saved during training:
- Saved in `samples/` directory
- Generated every 10 epochs
- 64 samples per grid (8×8)

## Monitoring Training

### Loss Tracking

The notebook tracks:
- Generator loss
- Discriminator loss
- Gradient norms (optional)
- Discriminator scores (optional)

### Visual Monitoring

- Loss plots are generated after training
- Sample images are displayed during training
- Checkpoint images saved periodically

## Key Concepts

### DCGAN Principles
- Use of transposed convolutions for upsampling
- Batch normalization for stability
- LeakyReLU for discriminator
- No fully connected layers (except generator input)
- No BatchNorm in generator's final layer

### Enhancements in This Implementation
1. **Increased Generator Capacity**: 256 filters instead of 128
2. **Minibatch Standard Deviation**: Encourages texture detail
3. **Label Smoothing**: Improves generalization

## Author

**Oussama**
- Email: oussamaaqebli0628@gmail.com
- GitHub: [OussAQ](https://github.com/OussAQ)

## Acknowledgments

- DCGAN paper: Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" (arXiv:1511.06434)
- PyTorch documentation and community

## Future Improvements

- [ ] Progressive growing for higher resolution
- [ ] Perceptual loss using VGG features
- [ ] PatchGAN discriminator
- [ ] Conditional GAN for class-specific generation
- [ ] FID/IS score evaluation
- [ ] Web interface for image generation
