# Plant-disease-detection
Dataset - https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
Trained model link - https://drive.google.com/file/d/1mV1o4Z8vmofskwi588xSvtHtKFmSFcCC/view?usp=sharing

# **🌱 Plant Disease Detection using CNN**
This repository contains a Convolutional Neural Network (CNN)-based model for detecting plant diseases. The model has been trained on a labeled dataset of plant images.  

## **📥 How to Use the Trained Model**  
Since the trained model file (`trained_model.keras`) is too large to be uploaded directly to GitHub, it is stored on **Google Drive**. You need to **download and load it** before using it.

### **🔧 Setup & Install Dependencies**  
Before running any scripts, install the necessary Python libraries:  
```bash
pip install gdown tensorflow
```

## **📌 Downloading & Loading the Trained Model**  
To download and load the trained CNN model, follow these steps:

1. **Run the provided script (`load_model.py`) to download the model automatically:**
   ```bash
   python load_model.py
   ```
   This script downloads the **trained CNN model** (`trained_model.keras`) from Google Drive and saves it locally.

2. **Import and use the model in your Python code:**
   ```python
   from load_model import model
   

