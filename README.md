## 🧠 Pre-trained Model
Due to GitHub's file size limitations, the trained CNN model is hosted on Google Drive. 

1. **Download the model:** [Click here to download plant_model_38_FINAL_V2.h5](https://drive.google.com/file/d/1cNGegLOL76ZW8m2kmyh1hJvpKztrBJqh/view?usp=sharing)
2. **Setup:** Place the downloaded `.h5` file in the root directory of this project before running `app.py`.

# 🌿 CropGuard: Smart Farming Project

CropGuard is an AI-driven platform designed to classify 38 different types of plant leaf diseases using Deep Learning.

## 🚀 Features
* **High Accuracy:** 99% accuracy on the test dataset.
* **Background Removal:** Integrated `rembg` pipeline for cleaner leaf segmentation.
* **Real-time Diagnostics:** Fast inference (<1.5s) using TensorFlow/Keras.
* **Safety First:** Implemented a 65% confidence threshold to block non-leaf images.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow, Keras (CNN)
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, PIL

## 📸 Project Goals & Conclusion
