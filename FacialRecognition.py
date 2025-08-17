import os
import logging
import sqlite3
import joblib
import time
import socket

from cryptography.fernet import Fernet
from collections import Counter

import numpy as np
import cv2 as cv
import dlib
import face_recognition

from scipy.spatial import distance as dist
from fer import FER

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------
# Setup directories & logging
# ---------------------------
os.makedirs("database", exist_ok=True)
os.makedirs("encryption_key", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_path = os.path.join(os.getcwd(), "logs", "system.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logging.info("✅ Logging initialized to both file and console")

# ---------------------------
# FaceDatabase: encrypted storage, classifier/pipeline persistence
# ---------------------------
class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect("database/face_recogniton.db", check_same_thread=False)
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

    def add_user(self, name, user_id_input, face_encoding_arrays):
        """
        Add a user and multiple face encodings (list of 1D numpy arrays)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''INSERT INTO users (name, user_id) VALUES (?, ?)''',
                          (name, user_id_input))
            db_user_id = cursor.lastrowid

            for enc in face_encoding_arrays:
                if isinstance(enc, np.ndarray):
                    enc_bytes = enc.tobytes()
                else:
                    enc_bytes = np.array(enc).tobytes()
                encrypted_data = self.cipher.encrypt(enc_bytes)
                cursor.execute('''INSERT INTO face_encodings (user_id, encrypted_data) VALUES (?, ?)''',
                               (db_user_id, encrypted_data))

            self.conn.commit()
            logging.info(f"Added user {name} (user_id: {user_id_input}) with {len(face_encoding_arrays)} encodings")
            return True
        except sqlite3.IntegrityError:
            logging.error(f"User ID {user_id_input} already exists")
            return False
        except sqlite3.Error as e:
            logging.error(f"Database Error: {e}")
            self.conn.rollback()
            return False

    def get_all_encodings(self):
        """
        Returns list of tuples: (user_id_str, name, numpy_array_encoding)
        user_id_str corresponds to users.user_id column (a string identifier provided at registration)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT u.user_id, u.name, fe.encrypted_data
                            FROM users u
                            JOIN face_encodings fe ON u.id = fe.user_id
                            ORDER BY fe.id ASC''')

            rows = cursor.fetchall()
            results = []
            for row in rows:
                user_id_str = row[0]
                name = row[1]
                enc_bytes = self.cipher.decrypt(row[2])
                enc = np.frombuffer(enc_bytes, dtype=np.float64)
                results.append((user_id_str, name, enc))
            return results
        except sqlite3.Error as e:
            logging.error(f"Database Error in get_all_encodings: {e}")
            return []

    def log_access(self, user_id, success, ip_address, emotion="unknown"):
        try:
            cursor = self.conn.cursor()
            if user_id is None:
                cursor.execute('''INSERT INTO access_logs (user_id, success, ip_address, emotion)
                                VALUES (NULL, ?, ?, ?)''',
                            (success, ip_address, emotion))
            else:
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
            logging.info("Saved classifier and label encoder to database/")
        except Exception as e:
            logging.error(f"Error saving classifier: {e}")

    def load_classifier(self):
        try:
            if os.path.exists("database/classifier.pkl") and os.path.exists("database/label_encoder.pkl"):
                clf = joblib.load("database/classifier.pkl")
                encoder = joblib.load("database/label_encoder.pkl")
                logging.info("Loaded classifier and label encoder from database/")
                return clf, encoder
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
        return None, None

# ---------------------------
# FaceProcessor: capture & liveness
# ---------------------------
class FaceProcessor:
    def __init__(self):
        try:
            self.detector = dlib.get_frontal_face_detector()
            # adjust model path as needed (expects models/shape_predictor_68_face_landmarks.dat)
            model_path = os.path.join(os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.predictor = dlib.shape_predictor(model_path)
        except Exception as e:
            logging.error(f"Error initializing FaceProcessor: {e}")
            raise

    def capture_face(self, timeout_sec=10):
        try:
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open video device")

            start_time = time.time()
            captured_frame = None

            while time.time() - start_time < timeout_sec:
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
            return avg_ear > 0.20  # slightly relaxed threshold for blink detection
        except Exception as e:
            logging.error(f"Error in check_liveness: {e}")
            return False
    
    def temporal_liveness_check(self, duration_sec=3):
        """Capture short sequence and verify blink/micro-movement patterns."""
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Cannot open camera for temporal liveness.")
            return 0.0

        ear_series = []
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.detector(gray)
            if faces:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                left_eye = self._get_eye_points(landmarks, "left")
                right_eye = self._get_eye_points(landmarks, "right")
                ear_series.append((self._eye_aspect_ratio(left_eye) + self._eye_aspect_ratio(right_eye)) / 2.0)
            time.sleep(0.05)

        cap.release()

        if not ear_series:
            return 0.0

        blinks = sum(1 for i in range(1, len(ear_series)) if ear_series[i] < 0.20 < ear_series[i-1])
        blink_score = min(blinks / 2.0, 1.0)
        micro_variation = np.std(ear_series)
        motion_score = min(micro_variation / 0.05, 1.0)

        return 0.6 * blink_score + 0.4 * motion_score
    
    def _image_quality_score(self, image):
        """Return a quality score between 0 and 1."""
        import cv2 as cv
        import numpy as np

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sharpness = cv.Laplacian(gray, cv.CV_64F).var()
        brightness = np.mean(gray)

        # Normalize: sharpness (0-500+) and brightness (0-255)
        sharpness_score = min(sharpness / 300.0, 1.0)
        brightness_score = max(0, min(1, (brightness - 40) / 170.0))

        return 0.6 * sharpness_score + 0.4 * brightness_score



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

# ---------------------------
# Sentiment (emotion) analysis
# ---------------------------
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

# ---------------------------
# Adaptive Fuzzy Inference System
# ---------------------------
class AdaptiveFIS:
    def __init__(self):
        self.face_match = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "face_match")
        self.emotion_stability = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "emotion_stability")
        self.liveness_score = ctrl.Antecedent(np.arange(0, 1.01, 0.01), "liveness_score")

        self.auth_confidence = ctrl.Consequent(np.arange(0, 1.01, 0.01), "auth_confidence")

        self._setup_membership_function()
        self._setup_rules()

    def _setup_membership_function(self):
        self.face_match['low'] = fuzz.gaussmf(self.face_match.universe, 0.15, 0.1)
        self.face_match['medium'] = fuzz.gaussmf(self.face_match.universe, 0.5, 0.15)
        self.face_match['high'] = fuzz.gaussmf(self.face_match.universe, 0.85, 0.1)

        self.emotion_stability['low'] = fuzz.gaussmf(self.emotion_stability.universe, 0.2, 0.1)
        self.emotion_stability['medium'] = fuzz.gaussmf(self.emotion_stability.universe, 0.5, 0.15)
        self.emotion_stability['high'] = fuzz.gaussmf(self.emotion_stability.universe, 0.8, 0.1)

        self.liveness_score['low'] = fuzz.gaussmf(self.liveness_score.universe, 0.1, 0.08)
        self.liveness_score['medium'] = fuzz.gaussmf(self.liveness_score.universe, 0.5, 0.15)
        self.liveness_score['high'] = fuzz.gaussmf(self.liveness_score.universe, 0.9, 0.1)

        self.auth_confidence['reject'] = fuzz.trimf(self.auth_confidence.universe, [0, 0, 0.4])
        self.auth_confidence['review'] = fuzz.trimf(self.auth_confidence.universe, [0.3, 0.5, 0.7])
        self.auth_confidence['accept'] = fuzz.trimf(self.auth_confidence.universe, [0.6, 1, 1])

    def _setup_rules(self):
        rules = [
            ctrl.Rule(self.face_match['high'] & self.emotion_stability['high']
                      & self.liveness_score['high'], self.auth_confidence['accept']),

            ctrl.Rule(self.face_match['high'] & self.emotion_stability['medium']
                      & self.liveness_score['high'], self.auth_confidence['accept']),

            ctrl.Rule(self.face_match['high'] & self.emotion_stability['low']
                      & self.liveness_score['high'], self.auth_confidence['review']),

            ctrl.Rule(self.face_match['medium'] & self.emotion_stability['medium']
                      & self.liveness_score['medium'], self.auth_confidence['review']),

            ctrl.Rule(self.face_match['high'] & self.emotion_stability['medium']
                      & self.liveness_score['medium'], self.auth_confidence['review']),

            ctrl.Rule(self.face_match['low'], self.auth_confidence['reject']),

            ctrl.Rule(self.liveness_score['low'], self.auth_confidence['reject']),

            ctrl.Rule(self.emotion_stability['low'] & self.face_match['medium'],
                      self.auth_confidence['reject'])
        ]

        self.control_system = ctrl.ControlSystem(rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def evaluate(self, face_match, emotion_stability, liveness_score):
        self.simulator.input['face_match'] = max(0, min(1, float(face_match)))
        self.simulator.input['emotion_stability'] = max(0, min(1, float(emotion_stability)))
        self.simulator.input['liveness_score'] = max(0, min(1, float(liveness_score)))

        try:
            self.simulator.compute()
            confidence = float(self.simulator.output['auth_confidence'])

            if confidence > 0.7:
                decision = "accept"
            elif confidence > 0.4:
                decision = "review"
            else:
                decision = "reject"

            return confidence, decision

        except Exception as e:
            logging.error(f"Fuzzy evaluation error: {e}")
            return 0.0, "reject"
        
    def adapt_membership_functions(self, recent_results):
        if not recent_results:
            return
        successful = [r for r in recent_results if r['decision'] == 'accept' and r.get('ground_truth') == True]
        if not successful:
            return

        mean_face_match = np.mean([r['face_match'] for r in successful])
        mean_emotion = np.mean([r['emotion_stability'] for r in successful])
        mean_liveness = np.mean([r['liveness_score'] for r in successful])

        self.face_match['high'] = fuzz.gaussmf(self.face_match.universe, mean_face_match, 0.1)
        self.emotion_stability['high'] = fuzz.gaussmf(self.emotion_stability.universe, mean_emotion, 0.1)
        self.liveness_score['high'] = fuzz.gaussmf(self.liveness_score.universe, mean_liveness, 0.1)

        logging.info(f"Fuzzy sets adapted: FM={mean_face_match:.2f}, ES={mean_emotion:.2f}, LS={mean_liveness:.2f}")


    def plot_membership_functions(self):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))

        for term in self.face_match.terms:
            ax0.plot(self.face_match.universe, fuzz.interp_membership(
                self.face_match.universe,
                self.face_match[term].mf,
                self.face_match.universe
            ), label=term)
        ax0.set_title('Face Match Confidence')
        ax0.legend()

        for term in self.emotion_stability.terms:
            ax1.plot(self.emotion_stability.universe, fuzz.interp_membership(
                self.emotion_stability.universe,
                self.emotion_stability[term].mf,
                self.emotion_stability.universe
            ), label=term)
        ax1.set_title('Emotion Stability')
        ax1.legend()

        for term in self.liveness_score.terms:
            ax2.plot(self.liveness_score.universe, fuzz.interp_membership(
                self.liveness_score.universe,
                self.liveness_score[term].mf,
                self.liveness_score.universe
            ), label=term)
        ax2.set_title('Liveness Score')
        ax2.legend()

        for term in self.auth_confidence.terms:
            ax3.plot(self.auth_confidence.universe, fuzz.interp_membership(
                self.auth_confidence.universe,
                self.auth_confidence[term].mf,
                self.auth_confidence.universe
            ), label=term)
        ax3.set_title('Authentication Confidence')
        ax3.legend()

        plt.tight_layout()
        plt.show()

# ---------------------------
# High-level FaceAuthSystem (no feature reduction)
# ---------------------------
class FaceAuthSystem:
    def __init__(self):
        try:
            self.face_db = FaceDatabase()
            self.face_processor = FaceProcessor()
            self.sentiment = SentimentAnalyzer()
            self.fuzzy_system = AdaptiveFIS()

            # Known face info loaded from DB (raw embeddings)
            self.known_encodings = []
            self.known_names = []
            self.known_user_ids = []
            self.name_to_user_id = {}

            self._load_known_faces()
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
            logging.info(f"Loaded {len(self.known_encodings)} known encodings for {len(set(self.known_names))} users")
        except Exception as e:
            logging.error(f"Error loading known faces: {e}")

    def _image_quality_score(self, image):
        """Return a quality score between 0 and 1."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        sharpness = cv.Laplacian(gray, cv.CV_64F).var()
        brightness = np.mean(gray)
        # Normalize: sharpness (0-1000) and brightness (0-255)
        sharpness_score = min(sharpness / 500.0, 1.0)
        brightness_score = max(0, min(1, (brightness - 50) / 150.0))
        return 0.6 * sharpness_score + 0.4 * brightness_score


    def extract_features(self, image):
        quality = self.face_processor._image_quality_score(image)
        if quality < 0.25:
            logging.warning(f"Image quality too low ({quality:.2f}) — skipping encoding.")
            return None
        try:
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_image)
            return face_encodings[0] if face_encodings else None
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return None


    def train_classifier(self):
        """
        Train an SVM classifier on the raw encodings (no feature reduction)
        """
        try:
            records = self.face_db.get_all_encodings()
            if not records:
                print("No users in database to train classifier.")
                return

            X = []
            y = []
            for _, name, enc in records:
                try:
                    enc = np.array(enc).flatten()
                    X.append(enc)
                    y.append(name)
                except Exception as e:
                    logging.error(f"Error processing encoding for {name}: {e}")

            if not X:
                print("No encodings available for training.")
                return

            print(f"Total samples: {len(X)} | Labels: {len(y)}")

            if len(set(y)) < 2:
                print("At least two users are required to train the classifier.")
                logging.warning("Classifier training skipped: only one class available.")
                return

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)

            clf = SVC(kernel="linear", probability=True)
            clf.fit(X, y_encoded)

            self.face_db.save_classifier(clf, encoder)
            logging.info("Classifier trained and saved.")
            print("Classifier trained successfully.")
        except Exception as e:
            logging.error(f"Error training classifier: {e}")
            print("Classifier training failed.")

    def register_user(self):
        print("\n=== User Registration ===")
        name = input("Enter your name: ").strip()
        user_id_input = input("Enter your ID: ").strip()

        if not name or not user_id_input:
            print("Name and ID cannot be empty.")
            return

        # Capture multiple samples
        print("Look at the camera. Capturing 15 face images (may take ~20-30s)...")
        face_images = []
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera.")
            return

        count = 0
        start_time = time.time()
        target = 15
        while count < target and (time.time() - start_time) < 60:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.face_processor.detector(gray)
            if faces and self.face_processor.check_liveness(frame, faces[0]):
                face_images.append(frame.copy())
                count += 1
                print(f"Captured image {count}/{target}")
                time.sleep(0.5)

        cap.release()

        if len(face_images) < 2:
            print("Failed to capture enough valid images.")
            return

        # Extract encodings
        successful_encodings = []
        for img in face_images:
            features = self.extract_features(img)
            if features is not None:
                successful_encodings.append(np.array(features).flatten())
            else:
                print("Warning: Face not detected in one of the captured images. Skipping it.")

        if len(successful_encodings) < 2:
            print("Insufficient valid face samples captured (minimum 2 required).")
            return

        # Add to DB using FaceDatabase
        added = self.face_db.add_user(name, user_id_input, successful_encodings)
        if not added:
            print("Failed to add user to the database (maybe user_id exists).")
            return

        self._load_known_faces()
        # Retrain classifier automatically
        self.train_classifier()
        print("Registration Successful!!")

    def _calculate_emotion_stability(self, face_image):
        frames = self.face_processor.capture_frames_over_time(5)
        if not frames:
            return 0.5

        emotions = []
        for frame in frames:
            result = self.sentiment.detector.top_emotion(frame)
            if result:
                emotions.append(result[0])

        if not emotions:
            return 0.5

        counter = Counter(emotions)
        total = sum(counter.values())
        dominant_emotion, count = counter.most_common(1)[0]
        consistency = count / total

        if dominant_emotion in ["angry", "fear", "disgust"]:
            consistency *= 0.7

        return max(0, min(1, consistency))

    def _get_liveness_score(self, face_image):
        gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
        faces = self.face_processor.detector(gray)

        if not faces:
            return 0.0

        face = faces[0]

        try:
            landmarks = self.face_processor.predictor(gray, face)

            left_eye = self.face_processor._get_eye_points(landmarks, "left")
            right_eye = self.face_processor._get_eye_points(landmarks, "right")
            left_ear = self.face_processor._eye_aspect_ratio(left_eye)
            right_ear = self.face_processor._eye_aspect_ratio(right_eye)

            avg_ear = (left_ear + right_ear) / 2.0
            ear_score = min(avg_ear / 0.3, 1.0)

            mouth_points = [(landmarks.part(p).x, landmarks.part(p).y)
                             for p in range(48, 68)]
            mouth_ear = self._mouth_aspect_ratio(mouth_points)
            mouth_score = max(0, min(1, 1 - (mouth_ear / 0.5)))

            liveness_score = (ear_score * 0.6 + mouth_score * 0.4)
            return max(0, min(1, liveness_score))

        except Exception as e:
            logging.error(f"Liveness detection error: {e}")
            return 0.5

    def _mouth_aspect_ratio(self, mouth_points):
        A = dist.euclidean(mouth_points[2], mouth_points[10])
        B = dist.euclidean(mouth_points[4], mouth_points[8])
        C = dist.euclidean(mouth_points[0], mouth_points[6])

        return (A + B) / (2.0 * C)

    def authentication(self):
        print("\n=== Authentication ===")

        face_image = self.face_processor.capture_face()
        if face_image is None:
            print("Liveness check failed or no face captured.")
            return

        features = self.extract_features(face_image)
        if features is None:
            print("No face encoding detected.")
            return

        clf, encoder = self.face_db.load_classifier()
        if clf is None or encoder is None:
            print("Classifier is not trained yet.")
            return

        try:
            features_arr = np.array(features).flatten().reshape(1, -1)
            logging.info(f"Authentication: feature shape {features_arr.shape}")

            probabilities = clf.predict_proba(features_arr)
            predicted_class = np.argmax(probabilities)
            name = encoder.inverse_transform([predicted_class])[0]

            print(f"Predicted probabilities: {probabilities}")
            print(f"Predicted class index: {predicted_class}")
            print(f"Predicted label (name): {name}")

            max_prob = float(np.max(probabilities))
            # debug log for classifier calibration
            logging.info(f"Max predicted probability: {max_prob:.4f}")

            if max_prob < 0.35:  # slightly lowered threshold for initial testing
                print("Prediction confidence too low. Aborting authentication.")
                return

            emotion_stability = self._calculate_emotion_stability(face_image)
            liveness_score = self._get_liveness_score(face_image)

            temporal_liveness = self.face_processor.temporal_liveness_check(3)
            liveness_score = (liveness_score + temporal_liveness) / 2.0


            print(f"\nRaw Scores: ")
            print(f"Face Match: {max_prob:.2f}")
            print(f"Emotion Stability: {emotion_stability:.2f}")
            print(f"Liveness Score: {liveness_score:.2f}")

            confidence, decision = self.fuzzy_system.evaluate(
                face_match=max_prob,
                emotion_stability=emotion_stability,
                liveness_score=liveness_score
            )

            print(f"\nAuthentication Confidence: {confidence:.2f}")
            print(f"System Decision: {decision.upper()}")

            try:
                ip_address = socket.gethostbyname(socket.gethostname())
            except:
                ip_address = "127.0.0.1"

            emotion_result = self.sentiment.detector.top_emotion(face_image)
            emotion = emotion_result[0] if emotion_result else "unknown"
            print(f"Detected emotion: {emotion}")

            if decision == "reject":
                print("Authentication Failed.")
                self.face_db.log_access(
                    user_id=self.name_to_user_id.get(name),
                    success=False,
                    ip_address=ip_address,
                    emotion=emotion
                )
                return

            elif decision == "review":
                print("Secondary authentication required due to medium confidence.")
                otp = input("Enter OTP: ")
                if otp != "123456":
                    print("Authentication Failed.")
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

        except Exception as e:
            logging.error(f"Authentication error: {e}")
            print("Authentication Failed")

# ---------------------------
# CLI main
# ---------------------------
if __name__ == '__main__':
    try:
        system = FaceAuthSystem()

        while True:
            print("\nMain Menu:")
            print("1. Register New User")
            print("2. Authenticate User")
            print("3. Analyze Emotions Over Time")
            print("4. Visualize Fuzzy Membership Functions")
            print("5. Train Classifier (manual)")
            print("6. Exit")

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
                    system.fuzzy_system.plot_membership_functions()
                elif choice == "5":
                    system.train_classifier()
                elif choice == "6":
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