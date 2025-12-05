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

        # Run prediction on single frame
        results = model.predict(
            source=frame,
            save=False,
            show=False,  # Don't show window
            device="cpu",
            conf=0.80,
            verbose=False  # Less console output
        )
        
        # Filter boxes by confidence
        for r in results:
            r.boxes = r.boxes[r.boxes.conf > 0.80]
        
        # Display in Streamlit if placeholder provided
        if frame_placeholder is not None:
            annotated_frame = results[0].plot()
            frame_placeholder.image(annotated_frame, channels="BGR")
    
    cap.release()
    return results