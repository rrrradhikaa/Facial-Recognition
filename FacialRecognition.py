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
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib
from fer import FER
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# Create necessary directories
os.makedirs("database", exist_ok=True)
os.makedirs("encryption_key", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
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
                       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS face_encodings(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id INTEGER NOT NULL,
                       encrypted_data BLOB NOT NULL,
                       FOREIGN KEY (user_id) REFERENCES users(id))''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS access_logs(
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id INTEGER,
                       attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       success BOOLEAN,
                       ip_address TEXT,
                       emotion TEXT)''')

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
        except sqlite3.IntegrityError:
            logging.error(f"User ID {user_id_input} already exists")
            return False
        except sqlite3.Error as e:
            logging.error(f"Database Error: {e}")
            self.conn.rollback()
            return False

    def get_all_encodings(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT u.user_id, u.name, fe.encrypted_data
                            FROM users u
                            JOIN face_encodings fe ON u.id = fe.user_id''')
            return [(row[0], row[1], np.frombuffer(self.cipher.decrypt(row[2]), dtype=np.float64))
                    for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Database Error in get_all_encodings: {e}")
            return []

    def log_access(self, user_id, success, ip_address, emotion="unknown"):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO access_logs 
                           (user_id, success, ip_address, emotion) 
                           VALUES (?, ?, ?, ?)''',
                         (user_id, success, ip_address, emotion))
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database Error in log_access: {e}")

    def save_classifier(self, clf, encoder):
        try:
            joblib.dump(clf, "database/classifier.pkl")
            joblib.dump(encoder, "database/label_encoder.pkl")
        except Exception as e:
            logging.error(f"Error saving classifier: {e}")

    def load_classifier(self):
        try:
            if os.path.exists("database/classifier.pkl") and os.path.exists("database/label_encoder.pkl"):
                clf = joblib.load("database/classifier.pkl")
                encoder = joblib.load("database/label_encoder.pkl")
                return clf, encoder
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
        return None, None
    
    def save_pipeline(self, pipeline):
        try:
            joblib.dump(pipeline, "database/feature_pipeline.pkl")
        except Exception as e:
            logging.error(f"Error saving feature pipeline: {e}")

    def load_pipeline(self):
        try:
            if os.path.exists("database/feature_pipeline.pkl"):
                return joblib.load("databse/feature_pipeline.pkl")
        except Exception as e:
            logging.error(f"Error loading feature pipeline: {e}")
        return None


class FaceProcessor:
    def __init__(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            
            model_path = os.path.join(os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            logging.error(f"Error initializing FaceProcessor: {e}")
            raise

    def capture_face(self):
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open video device")

            start_time = time.time()
            captured_frame = None

            while time.time() - start_time < 10:
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = self.detector(gray)

                if faces and self.check_liveness(frame, faces[0]):
                    captured_frame = frame.copy()
                    break

            cap.release()
            return captured_frame
        except Exception as e:
            logging.error(f"Error in capture_face: {e}")
            if 'cap' in locals():
                cap.release()
            return None

    def capture_frames_over_time(self, duration_sec=60):
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open video device")

            frames = []
            start = time.time()

            while time.time() - start < duration_sec:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                time.sleep(1)

            cap.release()
            return frames
        except Exception as e:
            logging.error(f"Error in capture_frames_over_time: {e}")
            if 'cap' in locals():
                cap.release()
            return []

    def check_liveness(self, frame, face):
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, face)
            left_eye = self._get_eye_points(landmarks, "left")
            right_eye = self._get_eye_points(landmarks, "right")

            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0
            return avg_ear > 0.25
        except Exception as e:
            logging.error(f"Error in check_liveness: {e}")
            return False

    def _get_eye_points(self, landmarks, side):
        points = [36, 37, 38, 39, 40, 41] if side == "left" else [42, 43, 44, 45, 46, 47]
        return [(landmarks.part(p).x, landmarks.part(p).y) for p in points]

    def _eye_aspect_ratio(self, eye_points):
        try:
            A = dist.euclidean(eye_points[1], eye_points[5])
            B = dist.euclidean(eye_points[2], eye_points[4])
            C = dist.euclidean(eye_points[0], eye_points[3])
            return (A + B) / (2.0 * C)
        except Exception as e:
            logging.error(f"Error in _eye_aspect_ratio: {e}")
            return 0


class SentimentAnalyzer:
    def __init__(self):
        try:
            self.detector = FER(mtcnn=True)
        except Exception as e:
            logging.error(f"Error initializing SentimentAnalyzer: {e}")
            raise

    def analyze_emotion_trend(self, frames):
        if not frames:
            print("No frames captured for analysis.")
            return

        emotions = []
        for frame in frames:
            try:
                result = self.detector.top_emotion(frame)
                if result:
                    emotions.append(result[0])
            except Exception as e:
                logging.error(f"Error analyzing frame: {e}")

        if not emotions:
            print("No emotions detected.")
            return

        counter = Counter(emotions)
        total = sum(counter.values())
        percentages = {emo: round((count / total * 100), 2) for emo, count in counter.items()}

        self._plot_emotions(percentages)
        self._regression(emotions)

    def _regression(self, emotions):
        if not emotions:
            print("No emotions available for regression.")
            return
        
        print("\n=== Linear Regression Results ===")
        x = np.arange(len(emotions)).reshape(-1, 1)
        for emotion_type in set(emotions):
            y = np.array([1 if emo == emotion_type else 0 for emo in emotions])
            model = LinearRegression().fit(x, y)
            print(f"Emotion: {emotion_type:8s} | w1: {model.coef_[0]:.6f}, w0: {model.intercept_:.6f}")
        print("=================================")
        input("Press Enter to continue...")

    def _plot_emotions(self, percentages):
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(percentages.keys(), percentages.values())
            plt.title("Emotion distribution over time")
            plt.ylabel("Percentage (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error plotting emotions: {e}")


class FaceAuthSystem:
    def __init__(self):
        try:
            self.face_db = FaceDatabase()
            self.face_processor = FaceProcessor()
            self.sentiment = SentimentAnalyzer()
            self.known_encodings = []
            self.known_names = []
            self.known_user_ids = []
            self.name_to_user_id = {}
            self._load_known_faces()
            self.feature_pipeline = None
            self._load_feature_pipeline()
        except Exception as e:
            logging.error(f"Error initializing FaceAuthSystem: {e}")
            raise

    def _load_known_faces(self):
        try:
            records = self.face_db.get_all_encodings()
            self.known_encodings = [enc for _, _, enc in records]
            self.known_names = [name for _, name, _ in records]
            self.known_user_ids = [user_id for user_id, _, _ in records]
            self.name_to_user_id = {name: user_id for user_id, name, _ in records}
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")

    def _load_feature_pipeline(self):
        self.feature_pipeline = self.face_db.load_pipeline()
        if self.feature_pipeline is None and self.known_encodings:
            self._train_feature_pipeline()

    def _train_feature_pipeline(self):
        try:
            pipeline = Pipeline([
                ('feature_selection', SelectKBest(f_classif, k =100)),
                ('dimensionality_reduction', PCA(n_components = 0.95))
            ])

            if len(self.known_encodings) > 1:
                pipeline.fit(self.known_encodings, self.known_names)
                self.feature_pipeline = pipeline
                self.face_db.save_pipeline(pipeline)
                logging.info("Feature reduction pipeline trained and saved.")
            else:
                print("Atleast 2 users are needed to train the pipeline.")
        
        except Exception as e:
            logging.error(f"Error training feature pipeline: {e}")

    def extract_hybrid_features(self, image):
        try:
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            if not face_encodings:
                return None
            deep_feat = face_encodings[0]

            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            resized = cv.resize(gray, (128, 128))
            resized = np.clip(resized, 0, 255).astype(np.uint8)

            hog_feat = hog(resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            combined_feat = np.concatenate((deep_feat, hog_feat))

            if self.feature_pipeline is not None:
                combined_feat = self.feature_pipeline.transform([combined_feat])[0]
            
            return combined_feat
        except Exception as e:
            logging.error(f"Error extracting hybrid features: {e}")
            return None
        
    def train_classifier(self):
        try:
            records = self.face_db.get_all_encodings()
            if not records:
                print("No users in database to train classifier.")
                return

            X = [enc for _, _, enc in records]
            if self.feature_pipeline is not None:
                X = self.feature_pipeline.transform(X)

            y = [name for _, name, _ in records]

            if len(set(y)) < 2:
                print("At least two users are required to train the classifier.")
                logging.warning("Classifier training skipped: only one class available.")
                return

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            clf = SVC(kernel="linear", probability=True)
            clf.fit(X, y_encoded)
            self.face_db.save_classifier(clf, encoder)
            print("Classifier trained successfully.")
        except Exception as e:
            logging.error(f"Error training classifier: {e}")
            print("Classifier training failed.")

    def register_user(self):
        print("\n=== User Registration ===")
        name = input("Enter your name: ").strip()
        user_id = input("Enter your id: ").strip()

        if not name or not user_id:
            print("Name and ID cannot be empty.")
            return

        print("Please look at the camera.")
        face_image = self.face_processor.capture_face()
        if face_image is None:
            print("Failed to capture a valid face image.")
            return

        features = self.extract_hybrid_features(face_image)
        if features is None:
            print("No face detected in the image.")
            return

        if self.face_db.add_user(name, user_id, features):
            print("Registration Successful!!")
            self._load_known_faces()
            self._train_feature_pipeline()
            self.train_classifier()
        else:
            print("Registration failed")


    def authentication(self):
        print("\n=== Authentication ===")

        face_image = self.face_processor.capture_face()
        if face_image is None:
            print("Liveness check failed.")
            return

        features = self.extract_hybrid_features(face_image)
        if features is None:
            print("No face detected.")
            return

        clf, encoder = self.face_db.load_classifier()
        if clf is None or encoder is None:
            print("Classifier is not trained yet.")
            return

        try:
            prediction = clf.predict([features])
            name = encoder.inverse_transform(prediction)[0]
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            print("Authentication failed.")
            return

        try:
            ip_address = socket.gethostbyname(socket.gethostname())
        except:
            ip_address = "127.0.0.1"

        emotion_result = self.sentiment.detector.top_emotion(face_image)
        emotion = emotion_result[0] if emotion_result else "unknown"
        print(f"Detected emotion: {emotion}")

        if emotion in ["angry", "fear", "disgust"]:
            print("High stress detected. Secondary authentication required.")
            otp = input("Enter OTP sent to your email: ")
            if otp != "123456":
                print("Authentication failed.")
                self.face_db.log_access(
                    user_id=self.name_to_user_id.get(name),
                    success=False,
                    ip_address=ip_address,
                    emotion=emotion
                )
                return

        print(f"Welcome {name}")
        user_id = self.name_to_user_id.get(name)
        if user_id:
            self.face_db.log_access(user_id, True, ip_address, emotion)
        else:
            print("User ID not found for authenticated name.")
            logging.warning(f"No matching user ID for name: {name}")

if __name__ == '__main__':
    try:
        system = FaceAuthSystem()

        while True:
            print("\nMain Menu:")
            print("1. Register New User")
            print("2. Authenticate User")
            print("3. Analyze Emotions Over Time")
            print("4. Exit")

            choice = input("Select Option: ").strip()

            try:
                if choice == "1":
                    system.register_user()
                elif choice == "2":
                    system.authentication()
                elif choice == "3":
                    print("Capturing 60 seconds of emotional data...")
                    frames = system.face_processor.capture_frames_over_time(60)
                    system.sentiment.analyze_emotion_trend(frames)
                elif choice == "4":
                    break
                else:
                    print("Invalid Choice.")
            except Exception as e:
                logging.error(f"Menu operation error: {e}")
                print("An error occurred. Check logs for details")

        print("System shutting down...")
    except Exception as e:
        logging.error(f"System startup error: {e}")
        print("System failed to start. Check logs for details.")