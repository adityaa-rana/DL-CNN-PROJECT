# Potato Leaf Disease Detection using Deep Learning (CNN)

An end-to-end Deep Learning project for **automatic detection of potato leaf diseases** using **Convolutional Neural Networks (CNNs)** built with **TensorFlow & Keras**, and deployed using a **Flask web application**.

This system takes an image of a potato leaf as input and predicts the disease category, helping in early diagnosis and crop protection.

---

## Features

- ✅ End-to-end Deep Learning pipeline
- ✅ Custom CNN architecture (from scratch)
- ✅ Image preprocessing & data augmentation
- ✅ Early stopping & dropout to prevent overfitting
- ✅ Model training, evaluation & inference
- ✅ Flask-based backend for deployment
- ✅ Easy-to-use web interface

---

## Model Architecture

The CNN model is built using `tf.keras.Sequential` with the following components:

- Input layer with fixed image size
- On-the-fly image preprocessing
- Data augmentation for better generalization
- Multiple convolution + max-pooling blocks
- Fully connected layers for classification
- Softmax output for multi-class prediction

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNEL)),

    # Preprocessing
    resize_and_rescale,
    data_augmentation,

    # Convolutional layers
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    # Classification head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])


## PROJECT STRUCTURE
potato-leaf-disease-detection/
│
├── dataset/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
│
├── model/
│   └── potato_disease_model.h5
│
├── static/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── requirements.txt
└── README.md

## CLONE THE REPO
git clone https://github.com/your-username/potato-leaf-disease-detection.git
cd potato-leaf-disease-detection

## VIRTUAL ENV
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

## DEPENDENCIES
pip install -r requirements.txt

## REQUIREMENTS
tensorflow==2.13.0
numpy==1.24.3
pandas==1.5.3
opencv-python==4.8.1.78
flask==2.3.3
werkzeug==2.3.7
matplotlib==3.7.2
scikit-learn==1.3.0
