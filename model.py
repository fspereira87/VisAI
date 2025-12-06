from ultralytics import YOLO
import cv2

model = YOLO("model3.pt")


def predict_video(video_path, frame_placeholder=None):
    """
    Predict on video with optional Streamlit frame display
    """
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            save=False,
            show=False,  
            device="cpu",
            conf=confidence_level,
            verbose=False  
        )
        
        
        if frame_placeholder is not None:
            annotated_frame = results[0].plot()
            frame_placeholder.image(annotated_frame, channels="BGR")
    
    cap.release()
    return results