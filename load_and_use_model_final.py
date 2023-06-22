# load_and_use_model.py

import pickle
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define some parameters
img_width, img_height = 150, 150

def predict(image_path, model, le_material, le_productgroup, le_sub_productgroep, le_return_rate):
    img = load_img(image_path, target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    predictions = model.predict(x)
    predicted_material = le_material.inverse_transform([np.argmax(predictions[0])])
    predicted_productgroup = le_productgroup.inverse_transform([np.argmax(predictions[1])])
    predicted_sub_productgroep = le_sub_productgroep.inverse_transform([np.argmax(predictions[2])])
    try:
        predicted_return_rate = le_return_rate.inverse_transform([np.argmax(predictions[3])])
    except ValueError:
        predicted_return_rate = ['Unknown']
    return predicted_material[0], predicted_productgroup[0], predicted_sub_productgroep[0], predicted_return_rate[0]

# Load the model and the label encoder
model = load_model("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/model.h5")

with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_material.pkl", "rb") as f:
    le_material = pickle.load(f)

with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_productgroup.pkl", "rb") as f:
    le_productgroup = pickle.load(f)

with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_sub_productgroep.pkl", "rb") as f:
    le_sub_productgroep = pickle.load(f)

with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_return_rate.pkl", "rb") as f:
    le_return_rate = pickle.load(f)

# Load model's history
with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/history.pkl", "rb") as f:
    history = pickle.load(f)

# Streamlit part
st.set_page_config(page_title="Image Classifier", layout="wide")
st.title("Image Classifier")

left_column, middle_column, right_column = st.columns([1, 2, 1])

with left_column:
    st.subheader("About the Project")
    st.markdown("""
    This is an image classification application built with deep learning.
    The model was trained to predict the material, product group, sub product group, 
    and return rate of shoe images. It was trained on a dataset of shoe images with 
    known labels for these categories.
    """)
    

with middle_column:
    
    #st.image(r"C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/shoe_returns_banner.png", caption='Project Image')  # Insert your image path here
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image_path = os.path.join("temp.jpg")
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption='Uploaded Image.', use_column_width=True)

        st.write("")
        st.write("Making the prediction...")

        material, productgroup, sub_productgroep, return_rate = predict(image_path, model, le_material, le_productgroup, le_sub_productgroep, le_return_rate)

        if st.checkbox('Show prediction results'):
            st.write(f'The predicted material is: {material}')
            st.write(f'The predicted product group is: {productgroup}')
            st.write(f'The predicted sub product group is: {sub_productgroep}')
            st.write(f'The predicted return rate is: {return_rate}')

with right_column:
    st.subheader('Model Training Details:')
    st.write(f"")
    st.write(f"Training accuracy (Material): {history['dense_1_accuracy'][-1]}")
    st.write(f"Validation accuracy (Material): {history['val_dense_1_accuracy'][-1]}")
    st.write(f"Training accuracy (Product group): {history['dense_2_accuracy'][-1]}")
    st.write(f"Validation accuracy (Product group): {history['val_dense_2_accuracy'][-1]}")
    st.write(f"Training accuracy (Sub product group): {history['dense_3_accuracy'][-1]}")
    st.write(f"Validation accuracy (Sub product group): {history['val_dense_3_accuracy'][-1]}")
    st.write(f"Training accuracy (Return rate): {history['dense_4_accuracy'][-1]}")
    st.write(f"Validation accuracy (Return rate): {history['val_dense_4_accuracy'][-1]}")
    st.write(f"")

    st.subheader("Training Graphs (Accuracy):")
    st.markdown("Training Accuracy (Material)")
    st.line_chart(history['dense_1_accuracy'])
    st.markdown("Training Accuracy (Product group)")
    st.line_chart(history['dense_2_accuracy'])
    st.markdown("Training Accuracy (Sub product group)")
    st.line_chart(history['dense_3_accuracy'])
    st.markdown("Training Accuracy (Return rate)")
    st.line_chart(history['dense_4_accuracy'])
    
    st.subheader("Training Graphs (Loss):")
    st.markdown("Training Loss (Material)")
    st.line_chart(history['dense_1_loss'])
    st.markdown("Training Loss (Product group)")
    st.line_chart(history['dense_2_loss'])
    st.markdown("Training Loss (Sub product group)")
    st.line_chart(history['dense_3_loss'])
    st.markdown("Training Loss (Return rate)")
    st.line_chart(history['dense_4_loss'])
