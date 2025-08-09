import os
import cv2
import numpy as np
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import SVC

class FaceSystemEvaluator:
    def __init__(self):
        # Corrected path structure for your specific dataset location
        self.dataset_path = os.path.expanduser(
            "C:\\Users\\ramra\\.cache\\kagglehub\\datasets\\jessicali9530\\lfw-dataset\\versions\\4\\lfw-deepfunneled\\lfw-deepfunneled"
        )
        self.min_face_size = (100, 100)
        self.min_images_per_person = 10
        self.max_people = 30
        self.k_folds = 3
        self.target_size = (128, 128)
        self.system = None  # Will be initialized for hybrid evaluation

        # Verify the exact directory structure exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at: {self.dataset_path}\n"
                "Please verify the path exists and contains subdirectories for each person"
            )

    def validate_image(self, img_path):
        """Validate and preprocess face images with error handling"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img = cv2.resize(img, self.target_size)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use CNN model if available, otherwise fallback to HOG
            try:
                face_locs = face_recognition.face_locations(rgb, model="cnn")
            except:
                face_locs = face_recognition.face_locations(rgb, model="hog")
                
            if not face_locs:
                return None
                
            top, right, bottom, left = face_locs[0]
            face_img = img[top:bottom, left:right]
            return cv2.resize(face_img, self.target_size)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def load_filtered_dataset(self):
        """Load dataset with proper path handling"""
        print(f"\nLoading from: {self.dataset_path}")
        
        # First find all valid person directories
        valid_people = []
        for person in os.listdir(self.dataset_path):
            person_dir = os.path.join(self.dataset_path, person)
            if os.path.isdir(person_dir):
                image_count = len([
                    f for f in os.listdir(person_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
                if image_count >= self.min_images_per_person:
                    valid_people.append((person, image_count))
        
        # Sort by image count and select top N
        valid_people.sort(key=lambda x: -x[1])
        selected_people = [p[0] for p in valid_people[:self.max_people]]
        
        if not selected_people:
            raise ValueError("No valid person directories found with enough images")
        
        # Load images with progress bar
        faces, labels = [], []
        for person in tqdm(selected_people, desc="Loading dataset"):
            person_dir = os.path.join(self.dataset_path, person)
            image_files = [
                f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ][:self.min_images_per_person]
            
            for img_file in image_files:
                img_path = os.path.join(person_dir, img_file)
                face = self.validate_image(img_path)
                if face is not None:
                    faces.append(face)
                    labels.append(person)

        print(f"\nSuccessfully loaded {len(faces)} images of {len(set(labels))} people")
        return faces, labels

    def extract_fast_features(self, img):
        """Robust feature extraction with fallbacks"""
        try:
            # First try the small model
            small_img = cv2.resize(img, (64, 64))
            encodings = face_recognition.face_encodings(
                small_img,
                known_face_locations=[(0, 64, 64, 0)],
                model="small"
            )
            if encodings:
                return encodings[0]
            
            # Fallback to large model if small fails
            return face_recognition.face_encodings(
                cv2.resize(img, (160, 160)),
                model="large"
            )[0]
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None

    def extract_true_hybrid_features(self, img):
        """Robust hybrid feature extraction with detailed error handling"""
        try:
            if img is None:
                print("Error: Input image is None")
                return None

            # 1. Deep features
            try:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)
                
                if not face_locations:
                    print("No face detected in image")
                    return None
                    
                encodings = face_recognition.face_encodings(rgb, face_locations)
                if not encodings:
                    print("Face encoding failed")
                    return None
                    
                deep_feat = encodings[0]
            except Exception as e:
                print(f"Deep feature extraction failed: {str(e)}")
                return None

            # 2. HOG features
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (128, 128))
                hog_feat = hog(resized, pixels_per_cell=(8,8), cells_per_block=(2,2))
            except Exception as e:
                print(f"HOG feature extraction failed: {str(e)}")
                return None

            # 3. Combine features
            try:
                combined = np.concatenate((deep_feat, hog_feat))
            except Exception as e:
                print(f"Feature concatenation failed: {str(e)}")
                return None

            # 4. Apply PCA if available
            if hasattr(self, 'system') and hasattr(self.system, 'feature_pipeline') and self.system.feature_pipeline:
                try:
                    return self.system.feature_pipeline.transform([combined])[0]
                except Exception as e:
                    print(f"PCA transformation failed: {str(e)}")
                    return combined  # Return untransformed features if PCA fails
            return combined

        except Exception as e:
            print(f"Unexpected error in feature extraction: {str(e)}")
            return None

    def run_true_evaluation(self):
        """Evaluates your ACTUAL system with better error handling"""
        try:
            # Load your production system
            class FaceAuthSystem:
                def __init__(self):
                    # Initialize with default PCA (or your actual pipeline)
                    from sklearn.decomposition import PCA
                    self.feature_pipeline = PCA(n_components=128)  # Example PCA
            
            self.system = FaceAuthSystem()
            
            # Load data
            faces, labels = self.load_filtered_dataset()
            
            # Extract features
            features = []
            valid_labels = []
            failed_count = 0
            
            for i, (face, label) in enumerate(tqdm(zip(faces, labels), desc="Extracting HYBRID features")):
                feat = self.extract_true_hybrid_features(face)
                if feat is not None:
                    features.append(feat)
                    valid_labels.append(label)
                else:
                    failed_count += 1
                    # Save the problematic image for debugging
                    cv2.imwrite(f"failed_image_{i}.jpg", face)
            
            print(f"\nFeature extraction summary:")
            print(f"Successfully extracted: {len(features)}")
            print(f"Failed extractions: {failed_count}")
            
            if len(features) < 2:
                raise ValueError(f"Only {len(features)} valid samples found - need at least 2 for evaluation")
            
            # Prepare data
            X = np.array(features)
            y = LabelEncoder().fit_transform(valid_labels)
            
            # K-Fold evaluation
            skf = StratifiedKFold(n_splits=min(3, len(np.unique(y))) if len(np.unique(y)) > 1 else 2)
            clf = SVC(kernel='linear', probability=True)
            
            accuracies = []
            for train_idx, test_idx in skf.split(X, y):
                clf.fit(X[train_idx], y[train_idx])
                acc = clf.score(X[test_idx], y[test_idx])
                accuracies.append(acc)
            
            print("\n=== YOUR TRUE SYSTEM PERFORMANCE ===")
            print(f"Hybrid Feature Accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
            print(f"Feature Dimensions: {X.shape[1]}")
            
        except Exception as e:
            print(f"\nEvaluation failed: {str(e)}")
            print("Possible solutions:")
            print("1. Check if faces are properly detected in the images")
            print("2. Verify the image preprocessing steps")
            print("3. Ensure all required packages are properly installed")
            print("4. Check saved 'failed_image_*.jpg' files for debugging")

    def visualize_samples(self, faces, labels, n=9):
        """Display sample faces with proper error handling"""
        try:
            plt.figure(figsize=(12, 8))
            unique_labels = list(set(labels))
            for i in range(min(n, len(unique_labels))):
                # Find first occurrence of each label
                idx = next(j for j, label in enumerate(labels) if label == unique_labels[i])
                plt.subplot(3, 3, i+1)
                plt.imshow(cv2.cvtColor(faces[idx], cv2.COLOR_BGR2RGB))
                plt.title(unique_labels[i][:15])  # Truncate long names
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization error: {str(e)}")

    def run_fast_evaluation(self):
        """Complete evaluation pipeline with robust error handling"""
        try:
            # 1. Load data
            print("\n=== Starting Evaluation ===")
            faces, labels = self.load_filtered_dataset()
            self.visualize_samples(faces, labels)
            
            # 2. Extract features
            print("\nExtracting features...")
            features, valid_labels = [], []
            for face, label in tqdm(zip(faces, labels), total=len(faces), desc="Processing"):
                enc = self.extract_fast_features(face)
                if enc is not None:
                    features.append(enc)
                    valid_labels.append(label)
            
            if len(features) < self.k_folds:
                raise ValueError(
                    f"Only {len(features)} valid samples found - "
                    f"need at least {self.k_folds} for evaluation"
                )
            
            # 3. Prepare data
            X = np.array(features)
            le = LabelEncoder()
            y = le.fit_transform(valid_labels)
            
            # 4. Run evaluation
            print("\nRunning cross-validation...")
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
            
            accuracies = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                clf.fit(X[train_idx], y[train_idx])
                acc = clf.score(X[test_idx], y[test_idx])
                accuracies.append(acc)
                print(f"Fold {fold}/{self.k_folds} accuracy: {acc:.4f}")
            
            # 5. Show results
            print("\n=== Final Results ===")
            print(f"Mean accuracy: {np.mean(accuracies):.4f}")
            print(f"Standard deviation: {np.std(accuracies):.4f}")
            print(f"Best fold: {np.max(accuracies):.4f}")
            print(f"Worst fold: {np.min(accuracies):.4f}")
            print("="*30)

        except Exception as e:
            print(f"\n!!! Evaluation failed: {str(e)} !!!")

if __name__ == "__main__":
    try:
        evaluator = FaceSystemEvaluator()
        print("1. Fast Evaluation")
        print("2. True Hybrid Evaluation")
        choice = input("Select evaluation mode (1/2): ")
        
        if choice == "1":
            evaluator.run_fast_evaluation()
        elif choice == "2":
            evaluator.run_true_evaluation()
        else:
            print("Invalid choice. Please select 1 or 2.")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please check:")
        print(f"1. Dataset exists at: {os.path.expanduser('~/.cache/kagglehub/datasets/jessicali9530/lfw-dataset/versions/4/lfw-deepfunneled')}")
        print("2. Directory contains subfolders for each person")
        print("3. You have read permissions for the directory")