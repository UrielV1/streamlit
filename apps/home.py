import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def upload_image():
    image = Image.open('apps/logr_conf_mat.PNG')
    return image

def app():
    st.title('Autism prediction among infants')
    image = upload_image()
    st.image(image)
    st.subheader("We wish you excellent results and perfect health")


