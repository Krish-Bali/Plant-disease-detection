Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import gdown
... import os
... import tensorflow as tf
... 
... model_path = "trained_model.keras"
... gdrive_link = "https://drive.google.com/uc?id=1mV1o4Z8vmofskwi588xSvtHtKFmSFcCC"  # Update with your file ID
... 
... # Check if model exists, otherwise download
... if not os.path.exists(model_path):
...     print("Downloading model...")
...     gdown.download(gdrive_link, model_path, quiet=False)
... else:
...     print("Model already exists, skipping download.")
... 
... # Load the model
... model = tf.keras.models.load_model(model_path)
... print("Model loaded successfully!")
>>> [DEBUG ON]
>>> [DEBUG OFF]
