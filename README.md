# Image Classification of Breast Cancer Tissue Cells

This project was developed as part of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano (A.Y. 2025–2026), within a Kaggle-based challenge organized for the course participants.

## What this is about

This project focuses on image classification using deep learning techniques, in particular Convolutional Neural Networks and Vision Transformers. The aim was to classify breast cancer tissue images into four classes: **Luminal A**, **Luminal B**, **HER2(+)**, and **Triple Negative**. The distinctive characteristics of cancer cells under the microscope are related to small and subtle differences between the different classes, which greatly increases the complexity of the task. Therefore, we had to face several challenges, such as class imbalance and limited dataset size.

Our work started with the analysis and cleaning of the dataset, which consists of **691 RGB images of different resolutions**, each accompanied by a **binary mask** highlighting the region of interest. We then moved on to building a custom neural network trained from scratch, before finally exploring pretrained models using transfer learning and fine-tuning. The main objective was to build a reliable classifier despite the small dataset size and the complexity of the tissue images.

The complete description of the methodology, experiments, and results is available in the [project report](Project%20Report.pdf). This README provides a concise overview of the implementation choices and the main findings.

## Repository structure

The repository is organized as follows:

```text
.
├── Code/                        # Model implementations and preprocessing scripts
│    └── DatasetCleaning.ipynb
│    └── FineTunedConvNeXt.ipynb
│    └── FineTunedTransformer.ipynb
│    └── ResNetFromScratch.ipynb
├── Project Report.pdf           # Report for the AN2DL Second Challenge
└── README.md
```

The source code is available in the [`Code/`](Code/) directory. The dataset is not included in this repository due to the original licensing conditions.


## Challenges and Methodology

**Focusing on fine details**: After removing corrupted and duplicated images, we kept 581 (out of 691) images. Nevertheless, using the original images without modification, both the number of samples and the level of detail were insufficient for such a complex task. Therefore, we split all the images into **224 × 224 tiles**, keeping only those containing relevant information according to the masks. Each tile was assigned the label of its original image, and during inference, we combined tile predictions using hard voting to obtain the final image prediction.

**Class imbalance**: Besides the slight class imbalance of **Luminal B** and **HER2(+)** with respect to **Luminal A**, the main issue was that the **Triple Negative** class, associated with the most aggressive type of breast cancer, was severely undersampled.

**Architecture selection**: We first tested a custom ResNet trained from scratch and then switched to **ConvNeXt Large** and **Vision Transformer**, which are pretrained models. In this regard, by splitting the original images into tiles, we also managed to create images matching the input dimensions required by our pretrained models. Regarding the loss function, standard unweighted Cross-Entropy performed best on the test set compared to custom class weights and Focal Loss.

**Inference and fine-tuning**: when predicting on an image, we classified its tiles separately and used **hard voting** to pick the final class, which worked better than mask-weighted soft voting. For the Vision Transformer, fine-tuning the last transformer block with different learning rates for the backbone and the classification head helped adapt the model to the dataset.


## Results

The best-performing model was the Vision Transformer (ViT), which achieved an F1 score of **40.71%** on the test set. ConvNeXt Large achieved **36.65%**, while the ResNet trained from scratch achieved 34.36%.

Overall, using **pretrained models gave a clear boost** over training from scratch on this small dataset, and careful cleaning and tiling helped make the most of the available images.
