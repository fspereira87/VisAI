import streamlit as st
import tempfile
import cv2
from model import predict_video, model


st.set_page_config(page_title="Drone Vision: Object Detection", layout="wide")
st.title("Drone Vision: Object Detection")

with st.sidebar:
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        
    

if video_file:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video_file.read())
    video_path = temp.name

    #st.video(video_path)

    if st.button("Run Prediction"):
        st.write("Processing video, please wait...")
        with st.sidebar:
            st.write("![Your Awsome GIF](https://i.makeagif.com/media/10-08-2020/QMj7da.gif)")

        cap=cv2.VideoCapture(video_path)
        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="BGR")

        cap.release()
        st.success("Processing complete!")


            