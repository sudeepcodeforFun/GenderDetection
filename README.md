# GenderDetection
This repository contains a gender detection system using deep learning. The model classifies images as male or female based on facial features. Built with YOLO for person detection and a custom-trained gender classification model, the system can be integrated into surveillance and security applications.
This project implements a gender classification model using Convolutional Neural Networks (CNNs) with Keras and TensorFlow. The model is trained on a dataset of facial images categorized as "man" and "woman" to classify gender accurately.

# Dataset Preprocessing
Images are loaded from the dataset directory and resized to 96x96 pixels.

Each image is converted to a numerical array (img_to_array) and normalized to a [0,1] range by dividing pixel values by 255.
Labels are assigned as:
1 for "woman"
0 for "man"
The dataset is split into 80% training and 20% validation using train_test_split().

# Data Augmentation
To enhance model generalization, ImageDataGenerator is used to apply:
Random Rotations (up to 30 degrees)
Width and Height Shifts (up to 20%)
Shear Transformations
Zooming (up to 25%)
Horizontal Flipping
Model Architecture
The CNN architecture consists of:
Three convolutional layers with Batch Normalization, ReLU activation, and L2 regularization to reduce overfitting.
MaxPooling layers to reduce spatial dimensions.
Dropout layers (30% to 50%) to improve generalization.
Fully Connected (Dense) layers leading to a softmax activation for classification.

# Compilation & Training
The model is compiled using the Adam optimizer with a learning rate of 1e-3.
Binary Cross-Entropy loss function is used since it's a two-class classification problem.
Early Stopping and ReduceLROnPlateau callbacks help prevent overfitting by stopping training when validation loss stops improving.

# Training Results
After training for up to 100 epochs, the training history (loss & accuracy) is plotted and saved as optimized_plot.png.
Blue Line → Training Accuracy
Red Line → Validation Accuracy
Green Line → Training Loss
Orange Line → Validation Loss
This plot helps visualize the model’s performance over epochs.

# Saving the Model
The trained model is saved as "gender_detection_optimized.model", which can be loaded for future predictions
