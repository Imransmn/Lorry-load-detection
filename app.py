import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Path to your model in Google Drive
model_path = '/content/drive/MyDrive/Load_Detection_YoloV8/Load_Segmentation_YoloV8/results/load_detection_yolov8/weights/best.pt'

# Load your trained YOLOv8 model
model = YOLO(model_path)

# Streamlit app setup
st.title('Lorry Load Detection')
st.write("Upload an image or video to detect if a lorry has a load or no load.")

# File uploader
uploaded_file = st.file_uploader("Choose an image or video...", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])

if uploaded_file is not None:
    # Check if the uploaded file is an image or video
    is_image = uploaded_file.type.startswith('image')

    if is_image:
        # Process image
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        # Get prediction
        results = model(img_np)

        # Draw bounding boxes and labels on the image
        annotated_image = results[0].plot()

        # Display the image with predictions
        st.image(annotated_image, caption='Processed Image', use_column_width=True)

    else:
        # Process video
        temp_file = 'temp_video.mp4'
        with open(temp_file, 'wb') as f:
            f.write(uploaded_file.read())

        # Load the video
        cap = cv2.VideoCapture(temp_file)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get prediction
            results = model(frame)

            # Draw bounding boxes and labels on the frame
            annotated_frame = results[0].plot()

            # Display the video frame
            stframe.image(annotated_frame, channels="BGR")

        cap.release()

    st.success("Processing complete!")

else:
    st.warning("Please upload an image or video file.")
