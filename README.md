
# Emotion Analysis App

This Flask application performs emotion recognition on uploaded audio and video files. It utilizes the Wav2Vec2 model for audio emotion recognition and a pre-trained facial emotion detection model for video analysis.

## Features

- Audio emotion recognition using Wav2Vec2.
- Video emotion recognition using a facial emotion detection model.
- Simple and intuitive web interface for uploading and analyzing files.

## Installation

### Prerequisites

- Python 3.7+
- Virtual environment tool (optional but recommended)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-analysis-app.git
   cd emotion-analysis-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application:**
   ```bash
   python app.py
   ```

2. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

3. **Upload audio and video files for emotion analysis.**

## Project Structure

```
emotion-analysis-app/
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── .gitignore
├── app.py
├── facialemotionmodel.h5
├── facialemotionmodel.json
├── README.md
├── requirements.txt
└── venv/
```

## Contributing

Contributions are welcome! Please create a pull request or open an issue for any improvements or suggestions.

## License

This project is licensed under the MIT License.

## Author

Hamza Bayraktepe
