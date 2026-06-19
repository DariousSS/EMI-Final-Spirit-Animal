# Spirit Animal

An interactive AI art project.

In legends from around the world—such as Shamanic traditions and ancient Chinese rituals—owls are regarded as messengers connecting different worlds, and can be summoned through hand gestures. A VAE neural network generates images of owls in real time based on the audience’s hand movements captured by a webcam. Five visual scenes play in an automatic loop, each accompanied by ritualistic drumbeats. The machine thus becomes the presider over the ritual.

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
