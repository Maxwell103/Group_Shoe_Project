# train_and_save_model.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define some parameters
img_width, img_height = 150, 150

base_dirs = [
    'D:\Van Gastel Images\wetransfer_bunnies-jr-ss21-en-ss22_2023-03-31_1035',
    'D:\Van Gastel Images\wetransfer_bunnies-jr-aw21-en-aw22_2023-03-31_1036'
]

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(img_width, img_height))
            x = img_to_array(img)
            images.append(x)
    return images

return_rate_bins = [0, 10, 20, np.inf]  # For example: Low (0-10), Medium (10-20), High (20+)
return_rate_labels = ['Low', 'Medium', 'High']

def train_and_save_model():
    df = pd.read_excel('C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/data.xlsx')
    df = df.drop('status_order', axis=1)
    
    df['SalesQty'] = df['SizeOrderedQty'].apply(lambda x: max(x, 0))
    df['ReturnQty'] = df['SizeOrderedQty'].apply(lambda x: -min(x, 0))
    total_sales = df.groupby(['ItemDescr', 'size','material', 'productgroup', 'seizoen', 'BrandCode', 'sub_productgroep'])['SalesQty'].sum()
    total_returns = df.groupby(['ItemDescr', 'size','material', 'productgroup', 'seizoen', 'BrandCode', 'sub_productgroep'])['ReturnQty'].sum()
    
    return_percent = (total_returns / (total_sales + total_returns)) * 100
    return_percent.rename('return_percentage', inplace=True)
    
    df_grouped = pd.DataFrame(return_percent).reset_index()
    df_grouped['return_percentage'] = df_grouped['return_percentage'].round(2)
    
    df_grouped['return_rate_category'] = pd.cut(df_grouped['return_percentage'], bins=return_rate_bins, labels=return_rate_labels)
    
    df = df.dropna(subset=['ItemCode', 'material', 'productgroup', 'sub_productgroep'])

    itemcode_material_dict = dict(zip(df['ItemCode'], df['material']))
    itemcode_productgroup_dict = dict(zip(df['ItemCode'], df['productgroup']))
    itemcode_sub_productgroep_dict = dict(zip(df['ItemCode'], df['sub_productgroep']))
    itemcode_return_rate_dict = dict(zip(df_grouped['ItemDescr'], df_grouped['return_rate_category']))

    images_list = []
    material_labels_list = []
    productgroup_labels_list = []
    sub_productgroep_labels_list = []
    return_rate_labels_list = []
    
    for itemcode in itemcode_material_dict.keys():
        for base_dir in base_dirs:
            item_path = os.path.join(base_dir, itemcode)
            if os.path.isdir(item_path):
                images = load_images_from_folder(item_path)
                for image in images:
                    images_list.append(image)
                    material_labels_list.append(itemcode_material_dict[itemcode])
                    productgroup_labels_list.append(itemcode_productgroup_dict[itemcode])
                    sub_productgroep_labels_list.append(itemcode_sub_productgroep_dict[itemcode])
                    return_rate_labels_list.append(itemcode_return_rate_dict.get(itemcode, 'Medium'))  # Default to 'Medium' if not found

    images_np = np.array(images_list)
    material_labels_np = np.array(material_labels_list)
    productgroup_labels_np = np.array(productgroup_labels_list)
    sub_productgroep_labels_np = np.array(sub_productgroep_labels_list)
    return_rate_labels_np = np.array(return_rate_labels_list)

    le_material = LabelEncoder()
    material_labels_np_int = le_material.fit_transform(material_labels_np)
    material_labels_np_cat = to_categorical(material_labels_np_int)

    le_productgroup = LabelEncoder()
    productgroup_labels_np_int = le_productgroup.fit_transform(productgroup_labels_np)
    productgroup_labels_np_cat = to_categorical(productgroup_labels_np_int)

    le_sub_productgroep = LabelEncoder()
    sub_productgroep_labels_np_int = le_sub_productgroep.fit_transform(sub_productgroep_labels_np)
    sub_productgroep_labels_np_cat = to_categorical(sub_productgroep_labels_np_int)

    le_return_rate = LabelEncoder()
    return_rate_labels_np_int = le_return_rate.fit_transform(return_rate_labels_np)
    return_rate_labels_np_cat = to_categorical(return_rate_labels_np_int)

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_return_rate.pkl", "wb") as f:
        pickle.dump(le_return_rate, f)

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_material.pkl", "wb") as f:
        pickle.dump(le_material, f)

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_productgroup.pkl", "wb") as f:
        pickle.dump(le_productgroup, f)

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/le_sub_productgroep.pkl", "wb") as f:
        pickle.dump(le_sub_productgroep, f)

    X_train, X_test, y_train_material, y_test_material, y_train_productgroup, y_test_productgroup, y_train_sub_productgroep, y_test_sub_productgroep, y_train_return_rate, y_test_return_rate = train_test_split(images_np, material_labels_np_cat, productgroup_labels_np_cat, sub_productgroep_labels_np_cat, return_rate_labels_np_cat, test_size=0.2, random_state=42)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    base_model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5)
    ])

    material_output = Dense(len(le_material.classes_), activation='softmax')(base_model.output)
    productgroup_output = Dense(len(le_productgroup.classes_), activation='softmax')(base_model.output)
    sub_productgroep_output = Dense(len(le_sub_productgroep.classes_), activation='softmax')(base_model.output)
    return_rate_output = Dense(len(le_return_rate.classes_), activation='softmax')(base_model.output)

    model = Model(inputs=base_model.input, outputs=[material_output, productgroup_output, sub_productgroep_output, return_rate_output])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, [y_train_material, y_train_productgroup, y_train_sub_productgroep, y_train_return_rate], epochs=10, validation_data=(X_test, [y_test_material, y_test_productgroup, y_test_sub_productgroep, y_test_return_rate]))

    model.save("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/model.h5")

    with open("C:/Users/maxwe/Documents/Github repos/Shoes-Modelling-Max/weights/history.pkl", "wb") as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    train_and_save_model()
