# Experiment 1: Face Recognition Evaluation (Corrected)
import os
import cv2
import numpy as np
import face_recognition
from sklearn.model_selection import LeaveOneOut
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from FacialRecognition import FaceAuthSystem

class FaceRecognitionBenchmark:
    def __init__(self):
        self.possible_paths = [
            os.path.expanduser("~/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled"),
            os.path.join(os.getcwd(), "lfw-deepfunneled"),
            os.path.join(os.getcwd(), "lfw"),
            os.path.join(os.getcwd(), "data")
        ]
        self.min_face_size = (80, 80)
        self.custom_system = FaceAuthSystem()

    def find_dataset_path(self):
        for path in self.possible_paths:
            if os.path.exists(path):
                possible_sub = os.path.join(path, "lfw-deepfunneled")
                return possible_sub if os.path.exists(possible_sub) else path
        return None

    def load_dataset(self):
        data_path = self.find_dataset_path()
        if not data_path:
            raise FileNotFoundError(f"Dataset not found in: {self.possible_paths}")

        print(f"\nLoading from: {data_path}")
        faces, labels = [], []
        people_dirs = [d for d in os.listdir(data_path)
                       if os.path.isdir(os.path.join(data_path, d))][:30]

        for person in people_dirs:
            person_dir = os.path.join(data_path, person)
            images = [f for f in os.listdir(person_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]

            for img_file in images:
                img_path = os.path.join(person_dir, img_file)
                img = self.validate_image(img_path)
                if img is not None:
                    faces.append(img)
                    labels.append(person)

        # Filter out people with fewer than 2 images
        grouped = defaultdict(list)
        for img, label in zip(faces, labels):
            grouped[label].append(img)

        faces, labels = [], []
        for label, imgs in grouped.items():
            if len(imgs) >= 2:
                for img in imgs:
                    faces.append(img)
                    labels.append(label)

        print(f"\nLoaded {len(faces)} images of {len(set(labels))} people")
        print("Sample distribution:", Counter(labels).most_common(3))
        return faces, labels

    def validate_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.resize(img, (256, 256))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb)

        if not face_locs:
            return None

        top, right, bottom, left = face_locs[0]
        w, h = right - left, bottom - top
        if w < self.min_face_size[0] or h < self.min_face_size[1]:
            return None

        face_crop = img[top:bottom, left:right]
        face_crop = cv2.resize(face_crop, (160, 160))
        return face_crop

    def extract_facenet_features(self, img):
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            return encodings[0] if encodings else None
        except Exception:
            return None

    def evaluate_systems(self, faces, labels):
        print("\n=== Evaluation Results ===")

        facenet_feats = [self.extract_facenet_features(img) for img in faces]
        valid_facenet = [(f, l) for f, l in zip(facenet_feats, labels) if f is not None]
        if valid_facenet:
            self.loo_evaluation("FaceNet", *zip(*valid_facenet))

        custom_feats = [self.custom_system.extract_hybrid_features(img) for img in faces]
        valid_custom = [(f, l) for f, l in zip(custom_feats, labels) if f is not None]
        if valid_custom:
            self.loo_evaluation("YourSystem", *zip(*valid_custom))

        self.evaluate_lbph(faces, labels)

    def loo_evaluation(self, name, features, labels):
        correct = 0
        loo = LeaveOneOut()

        for train_idx, test_idx in loo.split(features):
            distances = [np.linalg.norm(features[test_idx[0]] - features[i]) for i in train_idx]
            pred = labels[train_idx[np.argmin(distances)]]
            if pred == labels[test_idx[0]]:
                correct += 1

        accuracy = correct / len(features)
        print(f"{name} Accuracy: {accuracy:.1%} ({len(features)} samples)")

    def evaluate_lbph(self, faces, labels):
        if len(faces) < 5:
            print("LBPH: Not enough samples")
            return

        gray_faces = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in faces]
        label_ids = {name: i for i, name in enumerate(set(labels))}
        y = np.array([label_ids[l] for l in labels])

        correct = 0
        loo = LeaveOneOut()
        for train_idx, test_idx in loo.split(gray_faces):
            model = cv2.face.LBPHFaceRecognizer_create()
            model.train([gray_faces[i] for i in train_idx], y[train_idx])
            pred_id, _ = model.predict(gray_faces[test_idx[0]])
            if pred_id == y[test_idx[0]]:
                correct += 1

        print(f"LBPH Accuracy: {correct/len(gray_faces):.1%} ({len(gray_faces)} samples)")



if __name__ == "__main__":
    benchmark = FaceRecognitionBenchmark()
    try:
        faces, labels = benchmark.load_dataset()

        plt.figure(figsize=(10, 6))
        for i in range(min(4, len(faces))):
            plt.subplot(2, 2, i + 1)
            plt.imshow(cv2.cvtColor(faces[i], cv2.COLOR_BGR2RGB))
            plt.title(labels[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        benchmark.evaluate_systems(faces, labels)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
