import streamlit as st
import tempfile
from model import model
import os


st.set_page_config(page_title="VisAI: Object Detection", layout="wide")
st.title("Drone Vision: Object Detection")

with st.sidebar:
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.80, 0.05)

if video_file:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(video_file.read())
    video_path = temp_input.name
    temp_input.close()

    if st.button("Run Detection"):
        with st.spinner("Processing video..."):
            
            results = model.predict(
                source=video_path,
                save=True,
                conf=confidence_threshold,
                project='runs/detect',
                name='output',
                exist_ok=True
            )
            
            
            output_path = 'runs/detect/output/video_path.avi' 
            
            st.success("Processing complete!")
            st.video(output_path)