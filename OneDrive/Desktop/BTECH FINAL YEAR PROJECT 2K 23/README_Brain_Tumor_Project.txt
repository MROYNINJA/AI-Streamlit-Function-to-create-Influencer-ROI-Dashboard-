Brain Tumor Detection Using Deep Learning

Project Overview
----------------
This project implements a Convolutional Neural Network (CNN) to detect brain tumors from medical imaging (MRI/CT) using deep learning. It uses image preprocessing, augmentation, and binary classification to distinguish between tumor and non-tumor images.

Key Features
------------
- Image preprocessing and augmentation
- CNN architecture with dropout for regularization
- Training and validation pipeline
- Accuracy and loss visualization

Requirements
------------
- Python 3.8+
- TensorFlow (with Keras)
- OpenCV (if used for any image handling outside Keras)
- NumPy
- Matplotlib
- scikit-learn (for extended evaluation, if needed)

Installation
------------
1. Clone or copy the project directory.
2. (Optional) Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate        (On Windows: venv\Scripts\activate)
3. Install dependencies:
   pip install tensorflow matplotlib numpy scikit-learn opencv-python

Directory Structure (expected)
------------------------------
project/
├── dataset/
│   ├── training_set/      (tumor and non-tumor subfolders)
│   └── test_set/          (tumor and non-tumor subfolders)
├── model.py               (main training script)
└── README.txt

Usage
-----
1. Organize your image dataset into the following folder format:
   dataset/training_set/tumor/
   dataset/training_set/non_tumor/
   dataset/test_set/tumor/
   dataset/test_set/non_tumor/

2. Run the training script (e.g., model.py):
   python model.py

Model Summary
-------------
- Input: 64x64 RGB images
- Layers:
  - Conv2D(32) + ReLU + MaxPooling
  - Conv2D(64) + ReLU + MaxPooling
  - Flatten
  - Dense(128) + ReLU + Dropout(0.5)
  - Dense(1) + Sigmoid (binary output)
- Loss: Binary Crossentropy
- Optimizer: Adam

Evaluation
----------
The script plots training and validation accuracy/loss. You may also compute:
- Precision, Recall, F1-score
- Confusion matrix
- ROC curve

Tips
----
- Use transfer learning or data augmentation for small datasets.
- Use a GPU if available for faster training.

Future Scope
------------
- Add tumor segmentation (e.g., U-Net)
- Classify tumor types (multi-class)
- Save/load trained models
- Build web/mobile UI for prediction

Contact
-------
Project Contributors: Sohon Mondal, Swati, Manas Roy, Anushka Sarkar, Surashree Nag
