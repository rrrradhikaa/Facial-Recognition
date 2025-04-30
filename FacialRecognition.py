import os
import logging
import sqlite3
from cryptography.fernet import Fernet
import numpy as np
import dlib
import cv2 as cv
import time
from scipy.spatial import distance as dist
import face_recognition
import socket

os.makedirs("database", exist_ok=True)
os.makedirs("encryption_key", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename="logs/system.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("database/face_recogniton.db")
        self._init_db()
        self._init_encryption()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       name TEXT NOT NULL,
                       user_id TEXT UNIQUE NOT NULL,
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP) ''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS face_encodings(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id INTEGER NOT NULL,
                       encrypted_data BLOB NOT NULL,
                       FOREIGN KEY (user_id) REFERENCES users(id)) ''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS access_logs(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id INTEGER,
                       attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       success BOOLEAN,
                       ip_address TEXT) ''')

        self.conn.commit()

    def _init_encryption(self):
        key_path = "encryption_key/encrypted.key"
        if not os.path.exists(key_path):
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)

        with open(key_path, 'rb') as f:
            self.cipher = Fernet(f.read())

    def add_user(self, name, user_id_input, face_encoding):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO users (name, user_id) VALUES (?, ?)''', 
                           (name, user_id_input))
            user_id = cursor.lastrowid

            encrypted_data = self.cipher.encrypt(face_encoding.tobytes())
            cursor.execute('''INSERT INTO face_encodings (user_id, encrypted_data) VALUES (?, ?)''',
                           (user_id, encrypted_data))

            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logging.error(f"Database Error: {e}")
            self.conn.rollback()
            return False

    def get_all_encodings(self):
        cursor = self.conn.cursor()
        cursor.execute('''SELECT u.user_id, u.name, fe.encrypted_data
                          FROM users u
                          JOIN face_encodings fe ON u.id = fe.user_id''')

        return [(row[0], row[1], np.frombuffer(self.cipher.decrypt(row[2]), dtype=np.float64))
                for row in cursor.fetchall()]

    def log_access(self, user_id, success, ip_address):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO access_logs (user_id, success, ip_address) VALUES (?, ?, ?)''',
                       (user_id, success, ip_address))

        self.conn.commit()

class FaceProcessor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:\\Users\\ramra\\Project\\models\\shape_predictor_68_face_landmarks.dat")

    def capture_face(self):
        cap = cv.VideoCapture(0)
        start_time = time.time()

        while time.time() - start_time < 10:
            ret, frame = cap.read()

            if not ret:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            if faces and self.check_liveness(frame, faces[0]):
                cap.release()
                return frame

        cap.release()
        return None

    def check_liveness(self, frame, face):
        landmarks = self.predictor(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), face)

        left_eye = self._get_eye_points(landmarks, "left")
        right_eye = self._get_eye_points(landmarks, "right")

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2.0
        return avg_ear > 0.25

    def _get_eye_points(self, landmarks, side):
        points = [36, 37, 38, 39, 40, 41] if side == "left" else [42, 43, 44, 45, 46, 47]
        return [(landmarks.part(p).x, landmarks.part(p).y) for p in points]

    def _eye_aspect_ratio(self, eye_points):
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])

        ear = (A + B) / (2.0 * C)
        return ear

    def extract_encodings(self, face_image):
        rgb_image = cv.cvtColor(face_image, cv.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        return encodings[0] if encodings else None

class FaceAuthSystem:
    def __init__(self):
        self.face_db = FaceDatabase()
        self.face_processor = FaceProcessor()
        self.known_encodings = []
        self.known_names = []
        self.known_user_ids = []
        self._load_known_faces()

    def _load_known_faces(self):
        records = self.face_db.get_all_encodings()
        self.known_encodings = [enc for _, _, enc in records]
        self.known_names = [name for _, name, _ in records]
        self.known_user_ids = [user_id for user_id, _, _ in records]

    def register_user(self):
        print("\n=== User Registration ===")
        name = input("Enter your name: ")
        user_id = input("Enter your id: ")

        print("Please look at the camera.")
        face_image = self.face_processor.capture_face()
        if face_image is None:
            print("Failed to capture a valid face image.")
            return

        encoding = self.face_processor.extract_encodings(face_image)
        if encoding is None:
            print("No face detected in the image.")
            return

        if self.face_db.add_user(name, user_id, encoding):
            print("Registration Successful!!")
            self._load_known_faces()
        else:
            print("Registration failed")

    def authentication(self):
        print("\n === Authentication ===")

        face_image = self.face_processor.capture_face()
        if face_image is None:
            print("Liveness check failed.")
            return

        encoding = self.face_processor.extract_encodings(face_image)
        if encoding is None:
            print("No face detected.")
            return

        matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=0.5)
        ip_address = socket.gethostbyname(socket.gethostname())

        

        if True in matches:
            match_idx = matches.index(True)
            name = self.known_names[match_idx]
            user_id = self.known_user_ids[match_idx]
            print(f"Welcome {name}")
            self.face_db.log_access(user_id, True, ip_address)

        else:
            print("Authentication failed.")
            self.face_db.log_access(None, False, ip_address)

if __name__ == '__main__':
    system = FaceAuthSystem()

    while True:
        print("\n Main Menu: ")
        print("1. Register New User")
        print("2. Authenticate User")
        print("3. Exit Program")

        choice = input("Select Option: ")

        try:
            if choice == "1":
                system.register_user()
            elif choice == "2":
                system.authentication()
            elif choice == "3":
                break
            else:
                print("Invalid Choice.")

        except Exception as e:
            logging.error(f"System Error: {e}")
            print("An error occurred. Check logs for details")

    print("System shutting down....")
