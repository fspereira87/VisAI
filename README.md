# VisAI

Technical README for the VisAI project: an object detection and multi-object tracking pipeline built around YOLOv11 and ByteTrack with a Streamlit UI for video input, processing, and output.

## Goals
- Real-time or batch object detection on video using YOLOv11 based model.
- Multi-object tracking across frames using ByteTrack.
- Simple web UI (Streamlit) for uploading, running, visualizing, and downloading processed video.
- Configurable thresholds and tracker parameters.

## Requirements
- Python 3.10+ (project used 3.12.10; adjust as needed)
- pip
- FFmpeg (for video encoding/decoding)
- Optional: CUDA-capable GPU + appropriate drivers for GPU acceleration

## Quick setup (macOS)
1. Clone repository
   ```bash
   git clone <repo-url> VisAI
   cd VisAI
   ```

2. Create and activate virtual environment
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install Python dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Install FFmpeg (Homebrew)
   ```bash
   brew install ffmpeg
   ```

## File layout
- app.py                 — Streamlit application and UI
- model.py               — Model loading and inference logic (YOLOv11 wrapper)
- tracker.py             — ByteTrack integration and tracking logic
- utils/                 — Utility modules (video I/O, drawing, IO)
- configs/bytetrack.yaml — ByteTrack configuration
- assets/                — Demo videos, sample inputs
- models/                 — Model weight files (not in VCS)
- requirements.txt       — Python dependencies
- packages.txt           — System packages list
- .devcontainer/         — Dev container config for VS Code

Adjust paths in code if your layout differs.

## Configuration
- Model selection: change model path in model.py (e.g., models/yolov11n.pt).
- Device: set device in model loader (cpu or cuda).
- Confidence, IoU, and tracker thresholds: exposed in UI and in configs/bytetrack.yaml.
- Output formats: MP4 (H.264) is default via FFmpeg/OpenCV.

## Running the app
With the virtual environment active:
```bash
streamlit run app.py
```
App default address: http://localhost:8501

For headless processing or scripts, run the provided processing entrypoints (e.g., model inference script) directly:
```bash
python model.py --input <video> --output <out.mp4> --device cpu --conf 0.25
```
(Refer to script help for available CLI flags.)

## Usage summary
- Use Streamlit UI to select demo video or upload custom file.
- Configure confidence threshold and tracker parameters in the sidebar.
- Start processing; progress and logs appear in the UI.
- Download annotated video and per-frame/aggregate JSON/CSV outputs if enabled.

## Output
- Annotated video with bounding boxes and track IDs.
- Optional per-frame detection/tracking JSON or CSV summary.
- Performance metrics printed to console/UI (FPS, total frames, processing time).

## Troubleshooting
- FFmpeg errors: ensure ffmpeg binary is installed and on PATH (`which ffmpeg`).
- GPU not used: confirm CUDA drivers and torch installed with CUDA support; set device to `cuda` in model loader.
- High memory usage: reduce input resolution or process frame subsamples; increase tracker buffer parameters carefully.

## Development notes
- Tests: add unit tests under tests/ to validate IO and core algorithms.
- Linting/formatting: use flake8/black as configured in devcontainer.
- Devcontainer: open repository in VS Code Dev Container to reproduce environment.

## Contributing
- Fork the repo, implement changes on feature branch, open PR with description and tests.
- Keep model weights and large binaries out of VCS. Use models/ and update .gitignore accordingly.

## License
- Code: 
- Models: follow original model licenses (Ultralytics, third-party).

## Training Data
VisDrone: https://github.com/VisDrone/VisDrone-Dataset

## Citations and Acknowledgement

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}}
