import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('"C:\Users\deepi\Downloads\Face-Mask-Detector-Using-OpenCV-and-Python-master\Face-Mask-Detector-Using-OpenCV-and-Python-master\mask_detector.model"')  

# Function to make predictions
def predict_mask(image):
    # Preprocess the image
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize the pixel values to [0, 1]
    image = np.reshape(image, (1, 128, 128, 3))

    # Make prediction
    prediction = model.predict(image)

    return prediction

def main():
    st.title("Face Mask Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = cv2.imread(uploaded_file.name)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            result = predict_mask(image)

            if result[0][0] > 0.5:
                st.success("Person is wearing a mask!")
            else:
                st.error("Person is not wearing a mask!")

if __name__ == '__main__':
    main()
