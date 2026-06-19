# Spirit Animal

**Web blog:** https://app.notion.com/p/EMI-Final-Project-Blog-Spirit-Animal-en-38437ec6ecd38060907bcf9f7e95c3b0?source=copy_link

**Video:** https://drive.google.com/file/d/174pEpBLnKFyusvCWmI6JkcI-Hhgmm2al/view?usp=drive_link

---

## **Introduction**

I wanted to explore the sensation of machines becoming part of a ritual. Owls feature in many cultures; in shamanic traditions and on ancient Chinese ritual artefacts, they symbolise messengers connecting humans to the spiritual world. My aim was to use AI to ‘summon’ the image of an owl through gestures, allowing the audience to feel as though they are participating in a digital ritual, rather than merely interacting with a screen.

---

## **Related Technologies and Creative Works**

- VAE (Variational Autoencoder) image generation
- MediaPipe hand and facial tracking
- TouchDesigner real-time visuals
- Shamanic ritual aesthetics / ancient Chinese totemic imagery

---

## **Overview of the Design and Development Process**

I trained two VAE models, one for coloured owl faces and one for line art. These are connected to TouchDesigner via WebSocket. The position of the hand controls where the generated image appears on the screen. A feedback loop creates a trailing effect for the line art. The five visual scenes cycle automatically via an LFO approximately every 83 seconds, with a drumbeat triggered at each transition. 
The full process is documented on my webblog.

---

## **Summary of the Final Version**

The final piece uses a live camera feed as its base, overlaid with AI-generated owl images that respond to the audience’s gestures. The five scenes cycle automatically, with a drumbeat played upon each transition. 
Elements that worked well: hand tracking, VAE generation and scene transitions. Areas requiring improvement: if the Python server disconnects, the WebSocket connection sometimes needs to be manually restarted.

---

## **Evaluation**

The work largely met my expectations; during the final video recording, I performed in time with the music, hoping to give viewers a sense of participating in a ritual.

---

## **Reflections and Conclusions**

Through this project, I learnt how to train the system to generate attractive images using a small number of high-quality images. Initially, when I trained the model with over 200 images, the results were not pleasing. Ultimately, it was only after reducing the number of images and manually selecting each one that I achieved a result I was very satisfied with.

---

## **Repository Structure and Running Instructions**

**How to run:**

1. Open the terminal and start the server:
`cd gesture_classifier/serverpython serve.py`
2. Open `Spirit Animal.toe` in TouchDesigner
3. Enter full-screen presentation mode

---

## **Disclaimer regarding the use of AI tools**

Model: Claude Sonnet 4.6
Purpose: To assist with code generation
As it is not possible to generate a shareable link, this disclaimer is recorded here

---

## **Use of other third-party resources**

- **MediaPipe**: https://github.com/google-ai-edge/mediapipe
- **Freesound.org**: Drum sound effects used for scene transitions
