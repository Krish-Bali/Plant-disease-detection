import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os
import sys

# Ensure the default encoding is set to UTF-8
if sys.platform == "win32":
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')


# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    
    # Convert the uploaded file to a PIL Image
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to the model's expected input size

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "images.webp"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            st.snow()  # Show snow effect during prediction# Add your long running task here


            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Define your class names
            class_name = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
              'Blueberry healthy', 'Cherry (including sour) Powdery mildew', 
              'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot', 
              'Corn (maize) Common rust', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy', 
              'Grape Black rot', 'Grape Esca (Black Measles)', 'Grape Leaf blight (Isariopsis Leaf Spot)', 
              'Grape healthy', 'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot',
              'Peach healthy', 'Pepper, bell Bacterial spot', 'Pepper, bell healthy', 
              'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
              'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 
              'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 
              'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
              'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite', 
              'Tomato Target Spot', 'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus',
              'Tomato healthy']

            
            st.success(f"Model is predicting it's a {class_name[result_index]}")
