# Early Learning Assistant

**Early Learning Assistant** is a real-time interactive educational app built with Python. It uses your computer’s **camera** and **speakers** to guide users—especially young learners—through playful actions and tasks. The app uses **MediaPipe** to detect **faces** (eyes, nose, mouth, ears) and **hands**, enabling visual and gesture-based interaction.

---

## 🧠 How It Works

- 📷 Captures real-time video using your **computer's webcam**
- 🧍 Detects **facial landmarks** (eyes, ears, nose, mouth) using **MediaPipe Face Mesh**
- ✋ Detects **hand landmarks** using **MediaPipe Hands**
- 🔊 Uses **system speakers** for interactive audio feedback (if implemented in your `main.py`)
- 💡 Designed for use in **early learning** environments to promote visual interaction

---

## 🛠 Prerequisites

Make sure you have:

- A working **webcam** and **speaker**
- Python **3.7 or newer** installed  
  Check with:

  ```bash
  python --version

## Local Setup

- Enable Python virtual environment and install dependencies:
```commandline
> python3 -m venv env
> source env/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt


```

## Run the app:
```commandline
> python main.py
```

## Remarks:
You might be prompted to allow the camera, speaker access. Make sure to allow it for the app to work properly.