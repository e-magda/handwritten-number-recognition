import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
### IMPORT YOUR CUSTOM FUNCTIONS


### ADD SOME CONTEXT (TITLE, TEXT...)

# Specifying canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 20, 50, 30)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")

realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Creating a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=realtime_update,
    height=448,
    width=448,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result is not None and canvas_result.image_data is not None:
    img_data = canvas_result.image_data
    im = Image.fromarray(img_data, mode="RGBA")
    rgb_im = im.convert('RGB')
    rgb_im.save("image.jpg")
    
    ### ADD YOUR FUNCTIONS TO MAKE PREDICTION AND DISPLAY RESULTS
    

