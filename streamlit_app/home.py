import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from scripts import predict

st.header("Привет, это Гайк. Я сделал то, что человечество сделало еще 35 лет назад. Но всему свое время.")

b_color = "#000000"
bg_color = "#FFFFFF"

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    canvas_result = st_canvas(
        stroke_width=30,
        stroke_color=b_color,
        background_color=bg_color,
        height=300,
        width=300,
        key="canvas"
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA") 
        img = img.convert("L")
        
        img = Image.fromarray(255 - np.array(img))
        
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = img_array.reshape(28,28) 
        
        if np.sum(img_array) != 0:
            probs = predict(img_array)
            probs = probs.numpy() * 100  

            with col2:
                for i in range(5):
                    st.write(f"**{i}: {probs[i]:.2f}%**")
                    st.progress(int(probs[i]))

            with col3:
                for i in range(5, 10):
                    st.write(f"**{i}: {probs[i]:.2f}%**")
                    st.progress(int(probs[i]))
