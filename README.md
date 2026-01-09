# Insect Image Generation using DCGAN with WGAN-GP

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) with Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate realistic insect images.

## Overview

The model is trained on a dataset of insect images organized by categories (Ant, Bee, Beetle, Butterfly, Dragonfly, Fly, Grasshopper, Ladybug, Mosquito, Spider, Wasp).

## Architecture

- **Generator**: Uses transposed convolutions to upsample from a 100-dimensional noise vector to 64×64 RGB images
- **Discriminator (Critic)**: Uses strided convolutions to classify images as real or fake
- **Loss Function**: WGAN-GP with gradient penalty for stable training

## Installation

1. Clone the repository:
```bash
git clone https://github.com/OussAQ/miniproject_DCGAN.git
cd miniproject_DCGAN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook insect_gan_training.ipynb
```

2. Run all cells sequentially to:
   - Load and preprocess the insect image dataset
   - Initialize the Generator and Discriminator networks
   - Train the model
   - Generate sample images

## Hyperparameters

- Image Size: 64×64
- Batch Size: 64
- Noise Dimension: 100
- Learning Rate: 0.0002
- N_CRITIC: 5 (discriminator updates per generator update)
- Lambda GP: 10 (gradient penalty coefficient)
- Epochs: 100

## Project Structure

```
.
├── data/                    # Insect image dataset (organized by category)
├── insect_gan_training.ipynb # Main training notebook
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Model Checkpoints

Trained model checkpoints are saved in the `checkpoints/` directory every 10 epochs.

## Generated Samples

Sample images generated during training are saved in the `samples/` directory.

## Author

Oussama - oussamaaqebli0628@gmail.com

## License

This project is open source and available for educational purposes.

