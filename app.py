import streamlit as st
import tempfile
from model import predict_video, model


st.set_page_config(page_title="VisAI: Object Detection", layout="wide")
st.title("Drone Vision: Object Detection")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 350px !important;
    }
    </style>
    """, unsafe_allow_html=True,
)

with st.sidebar:
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp.write(video_file.read())
    video_path = temp.name
    temp.close()

    if st.button("Run Prediction"):
        st.write("Processing video, please wait...")
        with st.sidebar:
            st.write("![Your Awesome GIF](https://i.makeagif.com/media/10-08-2020/QMj7da.gif)")

        frame_placeholder = st.empty()
        
        # Call predict_video with frame placeholder
        predict_video(video_path, frame_placeholder)
        
        st.success("Processing complete!")