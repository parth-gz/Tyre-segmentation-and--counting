import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

#load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

#function to count tyres based on mask area
def count_tyres(results, min_area=200):
    if results.masks is None:
        return 0

    masks = results.masks.data.cpu().numpy()
    return sum(m.sum() >= min_area for m in masks)


st.title("Tyre Segmentation & Counting")

#image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    results = model(
        image_path,
        imgsz=640,
        conf=0.5,
        iou=0.5,
        device=0,
        verbose=False
    )[0]

    tyre_count = count_tyres(results)
    output_img = results.plot()[..., ::-1]

    #display results
    st.subheader(f"Tyres Detected: {tyre_count}")
    st.image(output_img, use_column_width=True)

    os.remove(image_path)
