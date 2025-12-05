from ultralytics import YOLO

model = YOLO("model3.pt")


def predict_video(video_path):
    results = model.track(
        source = video_path, 
        save=False,
        tracker="bytetrack.yaml",
        show=True,
        device='gpu',
        conf=0.80)
    
    for r in results:
        r.boxes = r.boxes[r.boxes.conf > 0.80]
    return results
