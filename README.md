# Facial Recognition-based Authentication System
This is a Python-based facial authentication system with encryption and liveness detection. It allows users to register and authenticate using their face, while storing their data securely with encrypted encodings.

# Features
## * Face-Based User Registration & Authentication
Robust registration and authentication using hybrid facial features (deep + HOG).

## * Liveness Detection via Eye and Mouth Aspect Ratio (EAR/MAR)
Prevents spoofing by detecting natural blink and mouth movements using facial landmarks.

## * Hybrid Feature Extraction
Combines face_recognition embeddings with HOG features for improved classifier accuracy.

## * Encrypted Storage of Facial Data
Face encodings are encrypted using Fernet symmetric encryption before being stored in the database.

## * SQLite3 Database Integration
Local storage of users, face encodings, and authentication logs in a structured relational schema.

## * Emotion Detection & Stability Analysis
Uses deep learning-based FER (Facial Emotion Recognition) for emotional trend tracking and stability scoring.

## * Authentication Decision Using Fuzzy Logic
Fuzzy Inference System combines liveness, emotion stability, and face match confidence to decide access.

## * Real-Time Webcam Integration
Supports real-time frame capture for both registration and authentication with user feedback.

## * Logging & Monitoring
Logs all major events including registration attempts, authentication decisions, and system errors.

## * Classifier & Feature Pipeline Persistence
Saves and loads trained classifier (SVM) and feature reduction pipeline (PCA + SelectKBest) for reuse.

## * Multiple User Support
Supports registration and authentication of multiple unique users with per-user data handling.

## * Modular and Extensible Codebase
Easily extendable to support features like OTP-based 2FA, server-side validation, or GUI interface.

# Technologies Used
* Python 3.x – Core programming language.

* OpenCV (cv2) – Real-time video capture, grayscale conversion, and preprocessing.

* Dlib – Facial landmark detection and shape prediction.

* face_recognition – High-level face encoding and matching interface built on dlib.

* FER – Facial emotion recognition using deep learning (with MTCNN for accurate face detection).

* SQLite3 – Lightweight relational database for storing user profiles, encodings, and access logs.

* cryptography (Fernet) – Secure encryption of biometric data at rest.

* scikit-learn – Includes SVM classifier, PCA, SelectKBest, and LabelEncoder for ML processing.

* scikit-fuzzy (skfuzzy) – Used to implement fuzzy logic-based authentication decision-making.

* numpy – Numerical operations, image data transformation, and vector math.

* scipy – For geometric computations like Euclidean distances in landmark-based EAR/MAR.

* joblib – Efficient saving and loading of trained ML models and pipelines.

* matplotlib – Visualizes emotion distribution and fuzzy membership functions.

* collections (Counter) – Frequency analysis of detected emotions for trend analysis.

* socket – Retrieves device’s IP address for authentication logging.

* os, logging, time, pickle – Standard libraries used for path handling, logs, system tasks, and serialization.
# How to Run
## 1. Clone the Repository
git clone https://github.com/rrrradhikaa/Facial-Recognition.git
cd Facial-Recognition
## 2. Install Dependencies
pip install -r requirements.txt
## 3. Download the Required Dlib Model
Download shape_predictor_68_face_landmarks.dat from the official Dlib model files.
Place it inside the models/ folder and update the path in the script if needed.

## 4. Run the Program
python FacialRecognition.py
# Functionality Overview
## * register_user()
* Initiates user registration by:

* Capturing multiple live face images via webcam.

* Performing liveness detection (blink/mouth movement) on each capture.

* Extracting hybrid facial features (deep + HOG).

* Encrypting each face encoding using Fernet encryption.

* Storing encrypted encodings along with user metadata (name, ID) in a local SQLite3 database.

* Retraining and persisting the feature reduction pipeline and SVM classifier for recognition.

## * authentication()
* Performs secure user login by:

* Capturing a live face image via webcam.

* Verifying liveness using eye and mouth aspect ratio.

* Extracting hybrid facial features and passing them through the trained feature pipeline.

* Using the SVM classifier to predict user identity and associated probability.

* Computing emotion stability via real-time emotion trend analysis.

* Feeding the face match confidence, emotion stability, and liveness score into a fuzzy logic system to determine final access decision.

* Optionally prompting for OTP if confidence is medium ("review").

* Logging the authentication attempt with:

* Timestamp

* Success status

* IP address

* Detected emotion

* Associated user ID (if matched)

# Author
rrrradhikaa

# License
This project is for educational purposes only. Use responsibly.