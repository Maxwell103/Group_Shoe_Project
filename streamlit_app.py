import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import streamlit as st

# Define some parameters
img_width, img_height = 150, 150

base_dirs = [
    'C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/bunnies-jr-ss21-en-ss22_2023-03-31_1035',
    'C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/wetransfer_bunnies-jr-aw21-en-aw22_2023-03-31_1036'
]

# Function to load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(img_width, img_height))
            x = img_to_array(img)
            images.append(x)
    return images

# Function to create model
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    df = pd.read_excel('C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/data.xlsx')
    df = df.drop('status_order', axis=1)
    df = df.dropna(subset=['ItemCode', 'productgroup'])
    itemcode_label_dict = dict(zip(df['ItemCode'], df['productgroup']))

    images_list = []
    labels_list = []
    
    for itemcode, current_label in itemcode_label_dict.items():
        for base_dir in base_dirs:
            item_path = os.path.join(base_dir, itemcode)
            if os.path.isdir(item_path):
                images = load_images_from_folder(item_path)
                for image in images:
                    images_list.append(image)
                    labels_list.append(current_label)
    
    images_np = np.array(images_list)
    labels_np = np.array(labels_list)

    le = LabelEncoder()
    labels_np_int = le.fit_transform(labels_np)

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_productgroup.pkl", "wb") as f:
        pickle.dump(le, f)

    labels_np_cat = to_categorical(labels_np_int)

    X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np_cat, test_size=0.2, random_state=42)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = create_model(len(le.classes_))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save_weights("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/productgroup.h5")
    
    return model, le

def predict(image_path, model, le):
    img = load_img(image_path, target_size=(img_width, img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    prediction = model.predict(x)
    predicted_class = le.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

st.title("Image Classifier")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

model = None
le = None

if uploaded_file is not None:
    image_path = os.path.join("temp.jpg")
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(image_path, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Making the prediction...")
    model, le = train_model()
    st.write("Model predicted successfully!")
    label = predict(image_path, model, le)
    st.write(f'The predicted class is: {label}')
