import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

url = "https://drive.google.com/file/d/1j14OAUMlP5Z0hz7pwsLBEvGjvAaTfdQQ/view?usp=sharing"
file_id = "1j14OAUMlP5Z0hz7pwsLBEvGjvAaTfdQQ"

model_path = "trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path,quiet=False)

def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page",["HOME","DISEASE RECOGNITION","ABOUT", "CONTACT"])
img= Image.open("./img1.jpg")
st.image(img)

if(app_mode=="HOME"):
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", unsafe_allow_html=True)
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
  
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        class_name = ['Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

