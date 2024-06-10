import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import tempfile
import os

# Define the function for zero padding
def zero_padding(array):
    threshold = 120
    if array.shape[0] > threshold:
        return array[:threshold, :, :]
    elif array.shape[0] < threshold:
        padding = ((0, threshold - array.shape[0]), (0, 0), (0, 0))
        return np.pad(array, padding, mode="constant", constant_values=0)
    else:
        return array

# Define the function to extract landmarks
def extract_landmarks(path: str):
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {path}.")
        return

    data = []

    with mp_pose as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = [
                    [point.x, point.y, point.z]
                    for point in results.pose_landmarks.landmark
                ]
                data.append(landmarks)

    cap.release()
    np_data = np.array(data)
    return np_data

# Load the model
model = tf.keras.models.load_model("model.h5")

# Define the map of classes
maps = {
    0: "Lateral Side Raises",
    1: "Curl",
    2: "Leg Raise",
    3: "Over-Head Press",
    4: "Push-Up",
    5: "Squat",
}

st.title("Video Upload and Processing")

# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Display file details
    st.write("Filename:", uploaded_file.name)
    st.write("File type:", uploaded_file.type)
    st.write("File size:", uploaded_file.size, "bytes")
    
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    
    # Process the video and extract landmarks
    landmarks = extract_landmarks(temp_file_path)
    
    if landmarks is not None:
        # Zero padding the landmarks
        x = zero_padding(landmarks)
        
        # Predict the class
        x1 = np.expand_dims(x, axis=0)
        predict = model.predict(x1)
        
        l_p = list(predict[0])
        confident = max(l_p)
        index = l_p.index(confident)
        
        st.write("Prediction Results:")
        st.write(f"Class: {maps[index]}")
        st.write(f"Confidence: {confident * 100:.2f} %")
        
        # Display the uploaded video
        st.video(temp_file_path)
