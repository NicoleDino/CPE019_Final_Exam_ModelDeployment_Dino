# Commands needed for deployment

import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('predict_weather(classification).h5')
  return model
model=load_model()
st.write("""
# Welcome to the Weather Classifier! 
# Check and predict the weather today!"""
)
file=st.file_uploader("Select any weather-related image from your device",type=["jpg","png"])

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(96,96)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("You may now upload your chosen image file (paste your weather photograph here)")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Cloudy', 'Rain', 'Shine', 'Sunrise']
    string="THE WEATHER IS : "+class_names[np.argmax(prediction)]
    st.success(string)
