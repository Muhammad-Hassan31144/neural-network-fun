import streamlit as st
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
from streamlit_drawable_canvas import st_canvas

# Define the URI for the Flask server
URI = 'http://127.0.0.1:5000'

# Streamlit app title
st.title('MNIST Digit Recognizer & Neural Network Visualizer')

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Digit Recognizer", "Neural Network Visualizer"])

if app_mode == "Digit Recognizer":


    # Main section for drawing
    st.markdown('## Draw a digit')
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=20,
        stroke_color="black",
        background_color="white",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Preprocess the image
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))
        img = ImageOps.invert(Image.fromarray(img))  # Invert the image
        img = np.array(img).astype('float32') / 255  # Normalize the image
        img = img.reshape(1, 28, 28, 1)  # Reshape for model input

        # Display the processed image
        st.image(img.reshape(28, 28), width=150)

        def get_prediction(img):
            try:
                response = requests.post(URI, json={'image': img.tolist()})
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
                if 'prediction' in data:
                    return np.array(data['prediction'][0])
                else:
                    st.error(f"Error: {data.get('error', 'Unknown error')}")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                return None

        # Send the image to the Flask server for prediction
        if st.button('Get Prediction'):
            prediction = get_prediction(img)
            while prediction is not None:
                max_prediction = np.argmax(prediction)
                if max_prediction <= 9:
                    st.write(f'Prediction: {max_prediction}')
                    break
                else:
                    prediction = get_prediction(img)

elif app_mode == "Neural Network Visualizer":
    st.sidebar.markdown('## Input Image')
    index = st.sidebar.slider('Select image index', 0, 9999, 0)
    
    if st.button('Get random prediction'):
        try:
            response = requests.post(URI, data={})
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            if 'prediction' in data and 'image' in data:
                preds = data['prediction']
                image = np.array(data['image']).reshape(28, 28)

                st.sidebar.image(image, width=150)

                for layer, p in enumerate(preds):
                    numbers = np.squeeze(np.array(p))
                    fig, ax = plt.subplots(figsize=(32, 4))

                    if layer == 2:
                        row = 1
                        col = 10
                    else:
                        row = 2
                        col = 16

                    for i, number in enumerate(numbers):
                        ax = plt.subplot(row, col, i + 1)
                        ax.imshow(number * np.ones((8, 8, 3)).astype('float32'))
                        ax.set_xticks([])
                        ax.set_yticks([])

                        if layer == 2:
                            ax.set_xlabel(str(i), fontsize=40)
                    plt.subplots_adjust(wspace=0.05, hspace=0.05)
                    plt.tight_layout()
                    st.text('Layer {}'.format(layer + 1))
                    st.pyplot(fig)
            else:
                st.error(f"Error: {data.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
