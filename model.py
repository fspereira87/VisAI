from ultralytics import YOLO

model = YOLO("best.pt")

def predict_video(video_path):
    results = model.predict(
        source = video_path, 
        save=False,
        show=True,
        device='gpu',
        conf=0.80)
    
    for r in results:
        r.boxes = r.boxes[r.boxes.conf > 0.80]
    return results
