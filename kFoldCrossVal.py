import os
import logging
import numpy as np
import joblib
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Import from your main module
from FacialRecognition import FaceDatabase  # Adjust the import path as needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data_and_model():
    """Load the trained model, encoder, and face encodings from the database"""
    try:
        # Load the classifier and label encoder
        if not os.path.exists("database/classifier.pkl") or not os.path.exists("database/label_encoder.pkl"):
            logging.error("Trained model not found. Please train the model first.")
            return None, None, None, None
        
        clf = joblib.load("database/classifier.pkl")
        encoder = joblib.load("database/label_encoder.pkl")
        
        # Initialize your actual FaceDatabase
        face_db = FaceDatabase()
        records = face_db.get_all_encodings()
        
        if not records:
            logging.error("No face encodings found in database.")
            return None, None, None, None
            
        X = []
        y = []
        for _, name, enc in records:
            X.append(np.array(enc).flatten())
            y.append(name)
            
        # Transform labels to encoded values
        y_encoded = encoder.transform(y)
        
        return X, y_encoded, clf, encoder
    
    except Exception as e:
        logging.error(f"Error loading data and model: {e}")
        return None, None, None, None

def run_kfold_validation(X, y, model, k=5):
    """Run K-Fold Cross Validation"""
    try:
        logging.info(f"\nRunning {k}-Fold Cross Validation...")
        
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Calculate accuracy for each fold
        scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        logging.info(f"Accuracy for each fold: {scores}")
        logging.info(f"Average Accuracy: {np.mean(scores):.2f}")
        
        # Generate predictions for classification report
        y_pred = cross_val_predict(model, X, y, cv=kf)
        logging.info("\nClassification Report:\n" + classification_report(y, y_pred))
        
        return np.mean(scores)
    
    except Exception as e:
        logging.error(f"Error during cross validation: {e}")
        return None

if __name__ == "__main__":
    # Load data and model
    X, y, model, encoder = load_data_and_model()
    
    if X is not None and y is not None and model is not None:
        # Run 5-fold cross validation
        avg_accuracy = run_kfold_validation(X, y, model, k=5)
        
        if avg_accuracy is not None:
            logging.info(f"\nFinal average accuracy: {avg_accuracy:.2f}")
    else:
        logging.error("Failed to load data or model. Cannot perform validation.")