# EE559 Hand Gesture Recognition Project

My project implements hand gesture classification using the LeapGestRecog dataset and my own personal dataset. The goal of my project was to evaluate three models of increasing complexity, a baseline logistic regression classifier, an L2-regularized version of that, and a Convolutional Neural Network, and see how they each hold up under both the controlled dataset and real-world webcam conditions

---
## Dataset
This project uses the LeapGestRecog dataset from Kaggle:
https://www.kaggle.com/datasets/gti-upm/leapgestrecog

After downloading, place the dataset in the project directory as: leapGestRecog/
Custom data (if used) should be added to the corresponding gesture folders.

## Setup
Install required libraries:

```bash
pip install numpy matplotlib pillow torch torchvision opencv-python scikit-learn

How to Run Models:
Baseline model: python baseline.py
L2-regularized model: python l2regbaseline.py
cnn model: python3 cnn.py

Notes:
1. Dataset and recordings are not included due to size limits
2. Make sure the dataset is placed correctly before running the scripts
3. Random seeds are fixed for reproducibility