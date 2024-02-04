import tensorflow as tf
import keras #
from keras.models import load_model #
from tensorflow import keras

#from tensorflow.keras.model import load_model
import streamlit as st
import numpy as np

st.header('Vegetable Classification Model')
model = load_model ('D:\hack\Datathon 2.0\Vegetable_classification\classify\Image_classification.keras')
data_cat = ['beetroot',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'corn',
 'eggplant',
 'garlic',
 'ginger',
 'lemon',
 'lettuce',
 'onion',
 'pear',
 'peas',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip']

img_height=180
img_width=180
image =st.text_input('Enter Image name','test1.jpg')

image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width =200)
st.write('Vegetable in image is ->  ' + data_cat[np.argmax(score)])
st.write('With confidence of '+ str(np.max(score)*100))