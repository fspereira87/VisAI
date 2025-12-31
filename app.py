import streamlit as st
import tempfile
from model import model
import os
import cv2

st.set_page_config(page_title="VisAI: Object Detection", layout="wide")
st.title("Drone Vision: Object Detection")

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        width: 380px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Demo Videos")
    col1, col2, col3 = st.columns(3)
    with col1:
        demo_btn1 = st.button("Demos 1: Desert Road", use_container_width=True)
    with col2:
        demo_btn2 = st.button("Demos 2: Night traffic", use_container_width=True)
    with col3:
        demo_btn3 = st.button("Demos 3: Military Sample", use_container_width=True)

    st.divider()
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    st.header("Detection Settings")
    confidence_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_labels = st.checkbox("Show Labels", True)
    show_confidence = st.checkbox("Show Confidence Scores", True)

if "demo_choice" not in st.session_state:
    st.session_state.demo_choice = None

if demo_btn1:
    st.session_state.demo_choice = "demo1"

if demo_btn2:
    st.session_state.demo_choice = "demo2"

if demo_btn3:
    st.session_state.demo_choice = "demo3"

video_path = None

if st.session_state.demo_choice == "demo1" and not video_file:
    with open("/Users/Fspereira_dev/VisAI/car-driving-down-desert-road.mp4", "rb") as f:
        demo_bytes = f.read()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(demo_bytes)
    temp_input.close()
    video_path = temp_input.name

elif st.session_state.demo_choice == "demo2" and not video_file:
    with open("/Users/Fspereira_dev/VisAI/spinning-down-between-buildings.mp4", "rb") as f:
        demo_bytes = f.read()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(demo_bytes)
    temp_input.close()
    video_path = temp_input.name

elif st.session_state.demo_choice == "demo3" and not video_file:
    with open("/Users/Fspereira_dev/Downloads/12762044-hd_1920_1080_60fps.mp4", "rb") as f:
        demo_bytes = f.read()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(demo_bytes)
    temp_input.close()
    video_path = temp_input.name

elif video_file:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.close()
    video_path = temp_input.name

if video_path:
    cap_info = cv2.VideoCapture(video_path)
    fps = cap_info.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_info.release()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Duration", f"{duration:.1f}s")
    col2.metric("FPS", f"{fps:.1f}")
    col3.metric("Frames", total_frames)
    col4.metric("Resolution", f"{width}x{height}")

    st.divider()

    if st.button("▶️ Run Prediction", use_container_width=True):
        st.write("Processing video, please wait...")
        with st.sidebar:
            st.write("![Your Awesome GIF](https://i.makeagif.com/media/10-08-2020/QMj7da.gif)")

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_output.name
        temp_output.close()

        avi_output = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(avi_output, fourcc, fps, (width, height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detected_objects = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                results = model.predict(source=frame, save=False, show=False, conf=confidence_level, verbose=False)

                for r in results:
                    detected_objects += len(r.boxes)

                annotated_frame = results[0].plot(labels=show_labels, conf=show_confidence)
                out.write(annotated_frame)

                if frame_count % 10 == 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            cap.release()
            out.release()
            progress_bar.progress(1.0)
            status_text.text(f"Processed {frame_count} frames")

        status_text.text("Converting video to MP4 (H.264)...")
        mp4_output = avi_output.replace(".avi", ".mp4")
        os.system(f"ffmpeg -y -i {avi_output} -vcodec libx264 -preset fast -crf 22 {mp4_output}")

        status_text.text("Preparing video to display...")
        st.success("Processing Complete!")

        if os.path.exists(mp4_output) and os.path.getsize(mp4_output) > 0:
            st.video(mp4_output)

            with open(mp4_output, "rb") as f:
                video_bytes = f.read()

            st.markdown(
                f"""
                ### Detected Summary:
                - Total Frames Processed: {frame_count}
                - Total Objects Detected: {detected_objects}
                """
            )

            st.download_button(
                label="Download Processed Video",
                data=video_bytes,
                file_name="detection.mp4",
                mime="video/mp4",
            )
        else:
            st.error("Processed video file is not available.")

        try:
            os.unlink(output_path)
            os.unlink(avi_output)
        except:
            pass
