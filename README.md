# Pneumonia Detection from Chest X-Rays

## 📌 Overview

This repository contains a complete deep learning pipeline for detecting pneumonia from chest X-ray images. The project handles binary classification (Normal vs. Pneumonia) and utilizes transfer learning with **InceptionResNetV2** to achieve high accuracy and generalization on medical imaging data.

All code for training, data processing, and evaluation is contained within the `Normal_Pneumonia_Xray.ipynb` notebook.

## ✨ Key Features & Architecture

* **Efficient Data Pipeline:** Utilizes TensorFlow's `preprocessing.image_dataset_from_directory` for memory-efficient image loading and batching directly from the directory structure.
* **Transfer Learning:** Leverages the **InceptionResNetV2** architecture as a powerful feature extractor, with its base weights frozen during initial training.
* **Custom Classification Head:** Includes sequentially built custom dense and dropout layers on top of the base model to tailor the network to this specific binary classification task.
* **Integrated Data Augmentation:** Employs data augmentation layers directly within the model architecture to artificially expand the training dataset and improve model robustness against variations in X-ray positioning.

## 🛠️ Optimization & Overfitting Reduction

To ensure the model generalizes well to unseen data and doesn't just memorize the training set, several techniques were implemented:

* **Fine-Tuning:** After initial training, the top layers of the base model were unfrozen for fine-tuning with a highly reduced learning rate.
* **Class Weights:** Addressed inherent class imbalances in the medical dataset by penalizing misclassifications of the minority class more heavily.
* **Early Stopping:** Monitored validation metrics to halt training automatically when performance stopped improving, capturing the model at its optimal state.

## 📊 Performance Metrics

The model demonstrates strong predictive capabilities, balancing precision and recall effectively:

* **Training F1-Score:** 0.94
* **Testing F1-Score:** 0.90

## 🚀 Getting Started

### Prerequisites

To run this notebook locally with GPU acceleration, a configured environment is recommended:

* Python 3.8+
* TensorFlow 2.x (with GPU support)
* WSL2 (if operating on Windows)
* Conda (recommended for environment management)
* Matplotlib & NumPy

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Pneumonia-Detection.git
cd Pneumonia-Detection

```


2. Set up your environment and install dependencies:
```bash
conda create -n tf-env python=3.9
conda activate tf-env
pip install tensorflow numpy matplotlib

```



## 💻 Usage & Pre-Trained Model

You don't need to retrain the model from scratch to test it. The fully trained and fine-tuned model (`.h5` or `.keras` format) is hosted on Google Drive.

1. **Download the Model:** [Click here to download the final model](https://drive.google.com/file/d/1Kp-PvWh_DYywaLwcoyk-s6JwKx5VWobd/view?usp=sharing)
2. **Load and Predict:** Place the downloaded model file in your project directory and load it using TensorFlow:

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('path_to_downloaded_model_file')

# The model is now ready for evaluation or inference on new X-ray images

```

## 📂 Project Structure

* `Normal_Pneumonia_Xray.ipynb` - The primary notebook containing the data pipeline, model building, training loop, and evaluation.
* *(Include instructions on where users should place the raw dataset if they wish to train it themselves)*.
