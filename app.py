import numpy as np
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie
import tensorflow as tf
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image, ImageOps
from tensorflow.keras.utils import img_to_array, load_img
from io import BytesIO
import base64

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('/home/danushkumar/Dk/dev/Autoencoder-image-colorization/img_colorize_autoencoder.hdf5')
    return model
model = load_model()


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    
def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/jpg;base64,{img_str}" download ="result.jpg">Download result</a>'
	return href
def image_preprocessing(image_data, model):
    size = (256, 256)   
    img1_color=[]
    # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img1 = img_to_array(image_data)
    img1 = resize(img1 ,(256,256))
    img1_color.append(img1)
    img1_color = np.array(img1_color, dtype=float)
    img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
    img1_color = img1_color.reshape(img1_color.shape+(1,))
    output1 = model.predict(img1_color)
    output1 = output1*128
    result = np.zeros((256, 256, 3))
    result[:,:,0] = img1_color[0][:,:,0]
    result[:,:,1:] = output1[0]
    img2 = lab2rgb(result)
    img2 = resize(img2 ,(height,width))
    return img2

lottie_hello = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_urbk83vw.json")

st.header("Black and white image colorizer !")
para = """<p style="font-family:Courier; color:Black; font-size: 17px;">Do you wanna see a magic by adding colours to your black and white images!
Drop down below...</p>"""
st.markdown(para, unsafe_allow_html=True)
st_lottie(
        lottie_hello,
        speed=1,
        reverse=True,
        loop=True,
        quality="low", # medium ; high# canvas
        height=400,
        width=400,
        key=None,
    )


st.header("Upload your Image here!")
given_file = st.file_uploader("Upload the jpg or png which should be colorized")
if given_file is None:
    st.text("Please upload an image file")
else:
    print(given_file.name)
    image = Image.open(given_file)


    a,b = st.columns(2)
    a.text("Given B/W image:")
    a.image(image, use_column_width=True)
    width, height = image.size
    img1_color=[]
    
    img2 = image_preprocessing(image,model)
    b.text("Colorized Image:")
    b.image(img2, use_column_width=True)
    
    im = Image.fromarray((img2 * 255).astype(np.uint8))

    b = BytesIO()
    im.save(b,format="jpeg")        

    btn = st.download_button(
        label="Download image",
        data=b,
        file_name="colorized-img.png",
        mime="image/png")

