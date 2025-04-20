# 🧠 Handwritten Equation Recognition App

This Streamlit app allows users to upload images of handwritten mathematical equations. The app uses a CNN-based symbol recognition system to detect individual characters and evaluate the final expression.

## 🚀 Features

- Upload handwritten equation images
- Detect and visualize individual characters
- Predict symbols using pre-trained CNN models
- Display recognized expression and evaluate the result

## 🛠 Tech Stack

- Python
- OpenCV (`opencv-python`)
- TensorFlow / Keras
- NumPy
- Matplotlib
- Streamlit

## 📁 File Structure
handwritten-equation-app/ ├── app.py ├── utils.py ├── models/ │ ├── CNNModel_0.json │ ├── CNNModel_0.weights.h5 │ └── ... up to CNNModel_9 ├── test_images/ ├── requirements.txt └── README.md
