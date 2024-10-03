# Deepfake Detection using ResNet34 and ViT

This project focuses on training and analyzing models to detect deepfake images using the Kaggle real vs fake deepfake dataset.

## Project Structure

- `train.ipynb`: Training notebook for ResNet34 and ViT Small models using FastAI.
- `attribute.ipynb`: Notebook for feature map extraction and attribution analysis.

## Model Training

Two models were trained on the dataset:

1. ResNet34: Achieved 91% accuracy at 8 epochs
2. ViT Small: Achieved 94.5% accuracy at 7 epochs

## Attribution Analysis

The `attribute.ipynb` notebook performs the following analyses:

1. Feature map extraction and visualization for ResNet34
2. Attribution methods applied to both models for fake and real images:
   - Integrated Gradients
   - Saliency
   - DeepLIFT (DeepSHAP)
   - Input X Gradient
   - Feature Ablation

## Key Findings

The attribution analysis revealed that in 7 out of 10 test images, the models were focusing on the sclera (white part) of the eyes to differentiate between real and fake images.

## Attribution Methods Overview

- **Integrated Gradients**: Assigns importance scores to input features by gradually changing a baseline input to the actual input.
- **Saliency**: Returns gradients with respect to inputs as a baseline approach.
- **DeepLIFT (DeepSHAP)**: Explains predictions by comparing neuron activations to a reference state, extended to approximate SHAP values.
- **Input X Gradient**: Multiplies input with the gradient with respect to input.
- **Feature Ablation**: Replaces input features with a baseline and computes the difference in output.

## Usage

1. Run `train.ipynb` to train the ResNet34 and ViT Small models.
2. Use `attribute.ipynb` to perform feature map extraction and attribution analysis on the trained models.