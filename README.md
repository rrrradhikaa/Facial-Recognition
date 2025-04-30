# Facial Recognition-based Authentication System

This is a Python-based facial authentication system with encryption and liveness detection. It allows users to register and authenticate using their face, while storing their data securely with encrypted encodings.

---

## Features
- Face-based user registration and login
- Liveness detection using Eye Aspect Ratio (to avoid spoofing)
- Secure storage of facial data using Fernet encryption
- SQLite3 database integration
- Uses Dlib, OpenCV, and Facial Recognition Libraries

---

## Technologies Used

- Python 3.x  
- OpenCV (`cv2`)  
- Dlib  
- `face_recognition` (built on top of dlib)  
- SQLite3  
- Fernet encryption (`cryptography`)  
- `scipy`, `numpy`, `socket`, and standard libraries  

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/rrrradhikaa/Facial-Recognition.git
cd Facial-Recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Required Dlib Model

Download `shape_predictor_68_face_landmarks.dat` from the official Dlib model files.  
Place it inside the `models/` folder and update the path in the script if needed.

### 4. Run the Program

```bash
python facial_recognition.py
```

---

## Functionality Overview

- `register_user()` – Captures a live face, verifies liveness, extracts face encodings, encrypts them, and stores in the database.  
- `authentication()` – Captures a live face, compares with stored encrypted encodings, and logs the result with IP address.  

---

## Project Structure

```
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
├── facial_recognition.py
├── requirements.txt
└── README.md
```

---

## Author

rrrradhikaa

---

## License

This project is for educational purposes only. Use responsibly..