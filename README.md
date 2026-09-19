# Sign to Speech: ISL Recognition Using MediaPipe and Pattern Matching

A browser-based system that translates Indian Sign Language (ISL) gestures into spoken **English and Tamil** in real time.

## 🎯 Overview
This project was built as a final-year academic project to make communication more accessible for the deaf and hard-of-hearing community. It captures hand gestures through a webcam, recognizes them using a rule-based pattern matcher, and converts them into spoken output — bilingually.

## 🏆 Key Result
- Achieved **100% classification accuracy** on the tested gesture set.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI
- **Real-time Communication:** WebSocket
- **Hand Tracking:** MediaPipe
- **Recognition Logic:** Rule-based five-element finger pattern matcher
- **Frontend:** HTML, JavaScript

## ⚙️ How It Works
1. The browser captures live hand gestures via webcam.
2. MediaPipe extracts hand landmark data.
3. A pattern-matching algorithm compares finger positions against known ISL gesture patterns.
4. The matched gesture is converted to text and spoken aloud in English and Tamil.

## 🚀 Running the Project
```bash
# Clone the repo
git clone https://github.com/PriyaRenganathan-dot/sign-to-speech-isl-recognition.git
cd sign-to-speech-isl-recognition

# Install dependencies
pip install -r requirements.txt

# Run the backend
python app/main.py   # adjust filename to your actual entry point
```
Then open the frontend HTML file in your browser to start using the app.

## 👥 Team
Built with two teammates under faculty guidance.

## 📄 Documentation
A detailed 60-page academic report and presentation accompany this project (see `Sign_Recognition_ML_Guide.docx`).
