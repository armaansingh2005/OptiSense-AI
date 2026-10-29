# OptiSense AI

**See Smarter. Detect Faster.**

OptiSense AI is a local-first computer-vision application being developed in
phases. The foundation, YOLO detection service, and uploaded-image workflow are
in place. Video, camera, tracking, analytics, and broader export workflows are
being implemented in later phases. The dashboard does not show fabricated
detections or performance metrics.

## Architecture

The project is organized to keep responsibilities separate:

- `app.py` and `components/` — Streamlit application and reusable UI.
- `optisense/` — detection, tracking, media, analytics, and export services.
- `config.py` — shared defaults and local runtime configuration.
- `scripts/` — developer and environment utilities.
- `tests/` — automated tests.
- `assets/`, `sample_media/`, and `outputs/` — static assets, optional demos,
  and generated results.

The inference stack uses Ultralytics YOLO and NumPy, supports CPU inference,
selects CUDA when PyTorch reports it available, and reuses the loaded model.
Model weights are not committed to this repository.

## Detection engine (Phase 2)

`optisense.detector.Detector` accepts a `uint8` image array with shape
`(height, width, 3)` and returns validated `Detection` records. Confidence and
IoU thresholds, image size, and optional class IDs are validated before
inference. The first "Load YOLO model" click initializes the model; Ultralytics
may download its default `yolo11n.pt` weights at that time. To use local weights,
set `OPTISENSE_MODEL_PATH` to an existing model file. Loading and inference
failures are reported rather than replaced with simulated detections.

Example:

```python
import cv2
from optisense.detector import DetectionSettings, Detector

detector = Detector()
frame = cv2.imread("image.jpg")
detections = detector.detect(frame, DetectionSettings(confidence=0.3))
for detection in detections:
    print(detection.to_dict())
```

## Image detection (Phase 3)

Open **Image Detection**, upload a JPG/JPEG, PNG, or WEBP file, choose confidence
and IoU thresholds and model input resolution, then select **Run detection**.
Uploads are limited by `OPTISENSE_MAX_UPLOAD_SIZE_MB` (200 MB by default), and
decoded images are additionally capped at 40 million pixels. The page shows the
original and annotated image, real inference time, category counts, per-box
confidence and coordinates, and a PNG download. Image decoding and inference
occur locally; uploaded files are not persisted by the application.

## Windows PowerShell setup

From the project directory:

```powershell
python --version
code .
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python .\scripts\check_environment.py
streamlit run .\app.py
```

Run the test suite with:

```powershell
python -m pytest
```

If PowerShell blocks virtual-environment activation, use the current terminal
session's execution-policy bypass:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The project targets a local Python environment. Package availability can vary
by Python version and platform; resolve dependency-installation errors in the
active virtual environment before launching the application.

## Configuration

Copy `.env.example` to `.env` only if you need to set local configuration.
`OPTISENSE_MODEL_PATH` optionally identifies locally available model weights;
`OPTISENSE_MAX_UPLOAD_SIZE_MB` sets the image upload-size limit. Phase 3
supports image upload, inference, annotation, and PNG export; video and camera
uploads are not connected yet.

## Model and licensing

The planned detector uses Ultralytics YOLO. Model weights are not bundled.
Before downloading or redistributing a model, review the current Ultralytics
software and model licensing terms and the terms applicable to the selected
weights.

## Privacy

The intended application processes media locally and does not require paid APIs
or external AI services. This foundation does not upload or process user media.

## Development status

- [x] Phase 1: project foundation and initial Streamlit shell.
- [x] Phase 2: model initialization, parameter validation, and detection
  normalization.
- [x] Phase 3: image validation, inference, annotated preview, counts, and PNG
  download.
- [ ] Phase 4 onward: video workflows, tracking, analytics,
  exports, tests, and complete documentation.

## Author

ARMAAN SINGH

