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
import skfuzzy as fuzz
from skfuzzy import control as ctrl

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
                            JOIN face_encodings fe ON u.id = fe.user_id
                            ORDER BY fe.id ASC''')

            return [(row[0], row[1], np.frombuffer(self.cipher.decrypt(row[2]), dtype=np.float64))
                    for row in cursor.fetchall()]
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

class IntervalType2FIS:
    def __init__(self):
        self.face_match = ctrl.Antecedent(np.arange(0, 1.01, 0.1), "face_match")
        self.emotion_stability = ctrl.Antecedent(np.arange(0, 1.01, 0.1), "emotion_stability")
        self.liveness_score = ctrl.Antecedent(np.arange(0, 1.01, 0.1), "liveness_score")

        self.auth_confidence = ctrl.Consequent(np.arange(0, 1.1, 0.1), "auth_confidence")

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
        self.simulator.input['face_match'] = max(0, min(1, face_match))
        self.simulator.input['emotion_stability'] = max(0, min(1, emotion_stability))
        self.simulator.input['liveness_score'] = max(0, min(1, liveness_score))

        try:
            self.simulator.compute()
            confidence = self.simulator.output['auth_confidence']

            if confidence > 0.7:
                decision = "accept"
            elif confidence > 0.4:
                decision = "review"
            else:
                decision = "reject"

            return confidence, decision
        
        except Exception as e:
            print(f"Fuzzy evaluation error: {e}")
            return 0.0, "reject"

class FaceAuthSystem:
    def __init__(self):
        try:
            self.face_db = FaceDatabase()
            self.face_processor = FaceProcessor()
            self.sentiment = SentimentAnalyzer()
            self.fuzzy_system = IntervalType2FIS()
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
                ('feature_selection', SelectKBest(f_classif, k=150)),
                ('dimensionality_reduction', PCA(n_components=min(50, len(self.known_encodings) - 1)))
            ])


            if len(self.known_encodings) > 1:
                print("Before reduction:", np.array(self.known_encodings).shape)
                X = [np.array(enc).flatten() for enc in self.known_encodings]
                reduced = pipeline.fit_transform(X, self.known_names)

                print("After reduction:", reduced.shape)

                self.feature_pipeline = pipeline
                self.face_db.save_pipeline(pipeline)
                logging.info("Feature reduction pipeline trained and saved.")
            else:
                print("Atleast 2 users are needed to train the pipeline.")

        
        except Exception as e:
            logging.error(f"Error training feature pipeline: {e}")

    def extract_hybrid_features(self, image, apply_pipeline=True):
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

            if apply_pipeline and self.feature_pipeline is not None:
                combined_feat = self.feature_pipeline.transform([combined_feat])
                return combined_feat[0]  # Return flattened reduced features

            return combined_feat  # Raw features
        except Exception as e:
            logging.error(f"Error extracting hybrid features: {e}")
            return None

        
    def train_classifier(self):
        try:
            records = self.face_db.get_all_encodings()
            if not records:
                print("No users in database to train classifier.")
                return

            print("Preparing classifier training data:")
            X = []
            y = []
            for _, name, enc in records:
                try:
                    enc = np.array(enc).flatten()
                    X.append(enc)
                    y.append(name)
                    print(f" - {name} ({len(enc)} features)")
                except Exception as e:
                    logging.error(f"Error processing encoding for {name}: {e}")

            if self.feature_pipeline is not None:
                X = self.feature_pipeline.transform(X)

            print(f"Total samples: {len(X)} | Labels: {len(y)}")


            if len(set(y)) < 2:
                print("At least two users are required to train the classifier.")
                logging.warning("Classifier training skipped: only one class available.")
                return

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            print("Label mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))
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
        user_id_input = input("Enter your ID: ").strip()

        if not name or not user_id_input:
            print("Name and ID cannot be empty.")
            return

        # Try to add user only once
        try:
            cursor = self.face_db.conn.cursor()
            cursor.execute("INSERT INTO users (name, user_id) VALUES (?, ?)", (name, user_id_input))
            user_id = cursor.lastrowid
            self.face_db.conn.commit()
        except sqlite3.IntegrityError:
            print("User ID already exists.")
            logging.error(f"User ID {user_id_input} already exists")
            return
        except sqlite3.Error as e:
            print("Database error.")
            logging.error(f"User registration DB error: {e}")
            return

        print("Look at the camera. Capturing 5 face images...")

        face_images = []
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open camera.")
            return

        count = 0
        start_time = time.time()
        while count < 5 and (time.time() - start_time) < 30:
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = self.face_processor.detector(gray)
            if faces and self.face_processor.check_liveness(frame, faces[0]):
                face_images.append(frame.copy())
                count += 1
                print(f"Captured image {count}/5")
                time.sleep(1)

        cap.release()

        if len(face_images) < 5:
            print("Failed to capture enough valid images.")
            return

        # Now insert encodings for the user
        for img in face_images:
            features = self.extract_hybrid_features(img, apply_pipeline = False)
            if features is None:
                print("Face not detected in one of the images.")
                return

            if isinstance(features, np.ndarray) and features.ndim > 1:
                features = features.flatten()
            else:
                features = np.array(features).flatten()

            encrypted_data = self.face_db.cipher.encrypt(features.tobytes())

            encrypted_data = self.face_db.cipher.encrypt(features.tobytes())
            try:
                cursor.execute("INSERT INTO face_encodings (user_id, encrypted_data) VALUES (?, ?)", (user_id, encrypted_data))
            except sqlite3.Error as e:
                print("Error inserting face encoding.")
                logging.error(f"Face encoding insert error: {e}")
                return

        self.face_db.conn.commit()
        print("Captured and stored all samples successfully.")

        self._load_known_faces()
        self._train_feature_pipeline()
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
        total = len(counter)
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
                             for p in range(48, 68) ]
            mouth_ear = self._mouth_aspect_ratio(mouth_points)
            mouth_score = max(0, min(1, 1 - (mouth_ear/0.5)))

            liveness_score = (ear_score * 0.6 + mouth_score * 0.4)
            return(max(0, min(1, liveness_score)))

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
            probabilities = clf.predict_proba([features])
            predicted_class = np.argmax(probabilities)
            name = encoder.inverse_transform([predicted_class])[0]

            print(f"Predicted probabilities: {probabilities}")
            print(f"Predicted class index: {np.argmax(probabilities)}")
            print(f"Predicted label (name): {name}")

            max_prob = np.max(probabilities)
            if max_prob < 0.55:
                print("Prediction confidence too low. Aborting authentication.")
                return

            

            emotion_stability = self._calculate_emotion_stability(face_image)
            liveness_score = self._get_liveness_score(face_image)

            print(f"\nRaw Scores: ")
            print(f"Face Match: {max_prob:.2f}")
            print(f"Emotion Stability: {emotion_stability:.2f}")
            print(f"Liveness Score: {liveness_score:.2f}")

            confidence, decision = self.fuzzy_system.evaluate(
                face_match = max_prob,
                emotion_stability = emotion_stability,
                liveness_score = liveness_score
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
                    user_id = self.name_to_user_id.get(name),
                    success = False,
                    ip_address = ip_address,
                    emotion = emotion
                )
                return
            
            elif decision == "review":
                print("Secondary authentication required due to medium confidence.")
                otp = input("Enter OTP: ")
                if otp != "123456":
                    print("Authentication Failed.")
                    self.face_db.log_access(
                        user_id = self.name_to_user_id.get(name),
                        success = False,
                        ip_address = ip_address,
                        emotion = emotion
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