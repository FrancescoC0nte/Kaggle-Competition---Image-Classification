# Image Classification of Breast Cancer Tissue Cells

Project for the AN2DL Second Challenge

## What this is about

This project focuses on image classification using deep learning techniques, in particular Convolutional Neural Networks and Vision Transformers. The goal is to classify breast cancer tissue cell images into four classes: **Luminal A**, **Luminal B**, **HER2(+)**, and **Triple Negative**. We had to address several challenges, including noisy or corrupted outliers, a limited dataset size, and class imbalance.

The dataset originally consists of 691 RGB images of different resolutions, each with a binary mask showing the region of interest. After removing outliers, we kept 581 images and split them into 224 x 224 tiles with enough foreground pixels. Each tile gets the label of its original image, and during inference we combine tile predictions using hard voting to get the final image prediction.

Our work starts from analyzing and cleaning the dataset, moves to a custom ResNet trained from scratch, and then explores pretrained models using transfer learning and fine-tuning. The main objective was to build a reliable classifier even with a small dataset and sensitive tissue images.

The complete description of the methodology, experiments, and results is available in the [project report](Project%20Report.pdf). This README provides a concise overview of the implementation choices and the main findings.

## Repository structure

The repository is organized as follows:

```text
.
├── Code/                        # Model implementations and preprocessing scripts
│    └── ...
├── Project Report.pdf           # Report for the AN2DL Second Challenge
└── README.md
```

The source code is available in the Code/ directory.

Challenges and Methodology
Data cleaning and tiling were essential because of outliers and the small number of images. We removed corrupted and duplicated images (such as ones with a "Shrek" overlay or green artifacts) and split the remaining images into 224 x 224 tiles with at least 200 foreground pixels, keeping tiles from the same original image in the same split to avoid data leakage. We also used a stratified split to handle class imbalance (especially for the smaller Triple Negative class).

Architecture selection: we first tested a custom ResNet trained from scratch, and then moved to pretrained models (ConvNeXt Large and Vision Transformer). For the loss function, standard unweighted Cross-Entropy worked best on the test set compared to custom class weights and Focal Loss. We also skipped data augmentations because flips, rotations, crops, and color changes hurt performance on these tissue images.

Inference and fine-tuning: when predicting on an image, we classified its tiles separately and used hard voting to pick the final class, which worked better than mask-weighted soft voting. For the Vision Transformer, fine-tuning the last transformer block with different learning rates for the backbone and the classification head helped adapt the model to the dataset.

Results
The best-performing model was the Vision Transformer (ViT), which achieved an F1 score of 40.71% on the test set. ConvNeXt Large achieved 36.65%, while the ResNet trained from scratch achieved 34.36%.

Overall, using pretrained models gave a clear boost over training from scratch on this small dataset, and careful cleaning and tiling helped make the most of the available images.
