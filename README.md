# Spirit Animal

An interactive AI art installation for *Engaging Machine Intelligence 2026*.

The owl — a messenger between worlds in shamanic traditions and ancient Chinese ritual — is summoned through gesture. A VAE neural network generates owl imagery in real time, driven by the viewer's hand movements captured via webcam. Five visual scenes cycle automatically, each accompanied by ceremonial drum sounds. The machine becomes the ritual host.

---

## How to Run

### Requirements
- Python 3.9+
- TouchDesigner 2025
- A webcam

### Install Python dependencies
```bash
pip install torch torchvision websockets mediapipe opencv-python pillow numpy
```

### Step 1 — Start the AI server
```bash
cd gesture_classifier/server
python serve.py
```
Keep this running in the background.

### Step 2 — Open TouchDesigner
Open `Spirit Animal.toe` in TouchDesigner.

### Step 3 — View the installation
Press **F1** to enter full-screen performance mode.

> Note: Model files (`vae_owl.pt`, `vae_sketch.pt`) are not included due to file size. Contact the author to obtain them.
