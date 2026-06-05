# heart-rate-detection-rppg

## Project Overview

This project is an eye-based heart-rate monitoring system built using rPPG-style signal extraction from a live webcam feed. It uses MediaPipe face mesh to detect the eyes, extracts a region of interest (ROI) around the eyes, computes a green-channel pulse signal, applies bandpass filtering, and estimates heart rate in beats per minute (BPM) using an FFT peak search.

## Key Features

- Real-time webcam heart-rate estimation
- MediaPipe face mesh for robust eye detection
- Eye ROI extraction for pulse-related signal capture
- Signal normalization and bandpass filtering
- FFT-based BPM and frequency estimation
- Live overlay dashboard with heart rate, frequency, and signal graph

## Project Structure

- `app.py` - main application loop that reads webcam frames, detects the face, extracts eye ROI, computes the pulse signal, and displays results.
- `heart_rate.py` - stores the rolling signal buffer and exposes the current signal.
- `utils/face_detection.py` - initializes MediaPipe FaceMesh and detects face landmarks.
- `utils/roi_extraction.py` - extracts the eye region from the face landmarks.
- `utils/signal_processing.py` - normalizes and bandpass-filters the extracted signal.
- `utils/bpm_calculator.py` - computes heart rate and dominant frequency using FFT.
- `requirements.txt` - exact Python dependency versions required to run the project.
- `static/style.css` - styling placeholder for associated UIs or web pages (if needed).
- `data/` - optional folder for recorded signal output or saved data.
- `graphs/` - optional folder for generated signal graphs or plots.

## How It Works

1. `app.py` opens the default webcam using OpenCV.
2. Each frame is converted to RGB and passed through `FaceDetector`.
3. If a face is detected, landmarks around both eyes are used to extract an eye ROI.
4. The mean green intensity of this ROI is appended to a rolling signal buffer.
5. Once enough data is collected, `SignalProcessor` normalizes and bandpass filters the signal between 0.75 and 3.0 Hz.
6. `BPMCalculator` computes the FFT of the filtered signal, finds the dominant frequency, and converts it to BPM.
7. The live frame shows the detected ROI, BPM, frequency, and a scrolling green signal graph.

## Dependencies

This repository uses exact dependency versions in `requirements.txt`:

- `opencv-python==4.11.0.86`
- `mediapipe==0.10.14`
- `numpy==1.26.4`
- `scipy==1.15.3`
- `matplotlib==3.10.3`
- `pandas==2.3.0`

## Installation

1. Create or activate a Python 3.12 virtual environment.
2. Install the project requirements:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the main application from the project root:

```bash
python app.py
```

The application will open a window titled `Heart Rate Monitor Dashboard`.

Controls:

- Press `q` to quit.

## Expected Output

When the webcam feed is active, the display shows:

- `Heart Rate: <value> BPM`
- `Frequency: <value> Hz`
- `Eye-Based Heart Monitor`
- A live signal graph of the green channel from the eye ROI

The output is dependent on camera quality, lighting, and subject stability. For best results, keep your face steady and well lit.

## Data and Output Files

- `data/` may be used to store recorded signal outputs or dataset files if expanded later.
- `graphs/` may be used to save generated plots or signal visualizations.

This project currently performs live analysis and does not require pre-loaded sample data to operate.

## Notes

- The measurement is approximate and best used as a research/demo tool rather than a medical-grade sensor.
- The algorithm uses eye-region photoplethysmography principles, capturing tiny pulse-related intensity variations in the green channel.
- Actual accuracy depends on environmental conditions and camera signal quality.
- For best stability, keep the face steady, ensure even lighting, and avoid sudden movements.


