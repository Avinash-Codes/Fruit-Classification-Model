# Fruit Classification Model

#### Video Demo: [Watch the Demo](https://youtu.be/XILlDz8C_os)

## Description

This project implements a robust fruit classification model that can accurately identify 262 different types of fruits using deep learning. With a large dataset of 225,639 images, the model is trained to deliver high accuracy in real-world applications, including agriculture, retail, and dietary analysis.

## Project Structure

```plaintext
project/
│
├── project.py               # Main script for initializing and running the project
│
├── model.py                 # Core script for training the model, with data loading, transformations, and training logic
│
├── test_project.py          # Script for testing and making predictions using the trained model
│
├── class_names.txt          # Text file containing the names of the 262 fruit classes
│
├── requirements.txt         # Python dependencies (e.g., PyTorch, Torchvision)
│
├── fruit_classifier_model.onnx # Exported ONNX model file for efficient inference
│
└── README.md                # Documentation and project overview (this file)
```

## Key Features

- **MobileNetV2 Architecture**: Uses MobileNetV2, a highly efficient and lightweight deep learning model, ideal for deployment on edge devices.
- **Data Augmentation**: Employs techniques like random rotation, resizing, and normalization to improve generalization and accuracy across different conditions.
- **GPU Support**: Optimized to run on GPU for faster training, making it scalable for large datasets.
- **ONNX Export**: The model is saved in ONNX format for cross-platform compatibility and easier deployment in various environments.
- **Easy Inference**: The test_project.py script allows for straightforward predictions on new images.

## Installation Instructions

### Clone the Repository:
```bash
git clone [Your Repository URL]
cd Fruit_Classification_Model
```

### Install Dependencies:
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

## Data Preparation

The dataset consists of over 225,000 images across 262 fruit classes. The images are preprocessed with the following steps:

- **Resizing**: All images are resized to 224x224 pixels to match MobileNetV2 input requirements.
- **Data Augmentation**: Random rotations and flips are applied to enhance model learning by creating variations.
- **Normalization**: Pixel values are normalized to align with the model's pre-trained parameters.
- **Dataset Split**: Training (80%) and validation (20%) sets.

## Future Improvements and Applications

### Future Enhancements
- **Additional Data**: Expanding the dataset for even greater accuracy.
- **Deployment**: Integrating the model into mobile or web applications for real-time fruit classification.

### Applications

1. **Agriculture**
   - Assisting farmers in identifying and categorizing crops.

2. **Retail**
   - Automated sorting and inventory management.

3. **Dietary Analysis**
   - Real-time tracking and classification of fruit types for health studies.
