# HeartJam

HeartJam is an interactive web app that turns your **heart rate** and **hand gestures** into a live music performance as form of self-expression.

## 🎮 Features

- Your **Fitbit** (or heart-rate source) controls the **tempo (BPM)**.
- Your **hand gestures** in front of the webcam:
  - **Cue / mute instruments** (keyboard, guitar, bass, percussion)
  - **Control global volume** with a single-finger “pointer” gesture
  - **Shuffle music themes** with the 🤟 sign (jazz, chill, house, rock)

## 🚀 Getting Started

### Prereqs

- **Node.js** (LTS) + npm or yarn
- **Python 3.9+**
- A webcam
- A heart-rate source that works with `FitbitConnector` (or a mock)

### Installation
1. Clone the repository
```bash
git clone [your-repo-url]
cd codejam25
```

2. Create + activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv/Scripts/activate
```

3. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### Running the application

1. Backend setup
```bash
python app.py
```

2. Frontend setup
```bash
npm install
npm start
```
