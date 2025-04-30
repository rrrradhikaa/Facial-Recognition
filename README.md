# Facial Recognition-based Authentication System

This is a Python-based facial authentication system with encryption and liveness detection. It allows users to register and authenticate using their face, while storing their data securely with encrypted encodings.

---

# Features

- Face-based user registration and login
- Liveness detection using Eye Aspect Ratio (to avoid spoofing)
- Secure storage of facial data using Fernet encryption
- SQLite3 database integration
- Access logs with IP address and timestamp
- Uses Dlib, OpenCV, and Face Recognition libraries

---

# Technologies Used

- Python 3.x
- OpenCV (`cv2`)
- Dlib
- `face_recognition` (built on top of dlib)
- SQLite3
- Fernet encryption (from `cryptography`)
- `scipy`, `numpy`, `socket`, and standard libraries

---

# How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/rrrradhikaa/Facial-Recognition.git
cd Facial-Recognition

2. **Install Dependencies**
pip install -r requirements.txt

3. **Download the Required Dlib Model**

Download shape_predictor_68_face_landmarks.dat from Dlib Model Files
Place it in the models/ folder and update the path if needed.

4. **Run the Program**
python facial_recognition.py 

# Functionality Overview
register_user() – Captures a live face, verifies liveness, extracts face encodings, encrypts them, and stores in DB.

authentication() – Captures a live face, compares with stored encrypted encodings, and logs result with IP.

# Project Structure
Facial-Recognition/
│
├── database/
│   └── face_recogniton.db
├── encryption_key/
│   └── encrypted.key
├── logs/
│   └── system.log
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── your_script.py
└── README.md

# Author
rrrradhikaa


# License
This project is for educational purposes only. Use responsibly.