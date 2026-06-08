import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import logging

# Setup Logger
logger = logging.getLogger("classifier_service")

class ClassifierService:
    # Target Classes
    CLASSES = ["Floral", "Geometric", "Tribal", "Temple Border", "Animal", "Paisley"]
    
    # Models cache
    _rf_model = None
    _svm_model = None
    _gb_model = None  # Gradient Boosting as standard/fallback for XGBoost
    _scaler = None
    
    @classmethod
    def initialize_models(cls):
        """Generates a synthetic motif feature dataset and trains RF, SVM, and GB classifiers."""
        logger.info("Initializing motif classification models...")
        
        # Features schema: [area, perimeter, circularity, texture_score, hu_0, hu_1, SIFT_kp]
        np.random.seed(42)
        X = []
        y = []
        
        # Generate representative dataset for the classes to enable classifier training
        for label_idx, label in enumerate(cls.CLASSES):
            # 100 samples per class
            for _ in range(120):
                if label == "Floral":
                    # High circularity, medium texture, medium SIFT
                    area = np.random.normal(5000, 1000)
                    perimeter = np.random.normal(280, 40)
                    circularity = np.random.uniform(0.7, 0.9)
                    texture = np.random.normal(12.0, 2.0)
                    hu0 = np.random.normal(2.5, 0.3)
                    hu1 = np.random.normal(6.0, 0.8)
                    sift = np.random.normal(40, 10)
                elif label == "Geometric":
                    # Low-medium circularity (polygons/stars), low texture, low SIFT
                    area = np.random.normal(4000, 800)
                    perimeter = np.random.normal(260, 30)
                    circularity = np.random.uniform(0.4, 0.65)
                    texture = np.random.normal(5.0, 1.0)
                    hu0 = np.random.normal(1.8, 0.2)
                    hu1 = np.random.normal(4.5, 0.5)
                    sift = np.random.normal(20, 5)
                elif label == "Tribal":
                    # Low circularity, high texture (sharp details), high SIFT
                    area = np.random.normal(6000, 1500)
                    perimeter = np.random.normal(400, 80)
                    circularity = np.random.uniform(0.2, 0.5)
                    texture = np.random.normal(20.0, 3.0)
                    hu0 = np.random.normal(3.5, 0.5)
                    hu1 = np.random.normal(8.0, 1.2)
                    sift = np.random.normal(85, 15)
                elif label == "Temple Border":
                    # Triangular shapes, specific hu moments, directional textures
                    area = np.random.normal(8000, 2000)
                    perimeter = np.random.normal(500, 100)
                    circularity = np.random.uniform(0.5, 0.6)
                    texture = np.random.normal(15.0, 2.5)
                    hu0 = np.random.normal(2.0, 0.2)
                    hu1 = np.random.normal(5.0, 0.6)
                    sift = np.random.normal(50, 10)
                elif label == "Animal":
                    # Complex geometry, high perimeter relative to area, moderate circularity
                    area = np.random.normal(9000, 2500)
                    perimeter = np.random.normal(600, 150)
                    circularity = np.random.uniform(0.3, 0.55)
                    texture = np.random.normal(14.0, 2.5)
                    hu0 = np.random.normal(3.2, 0.4)
                    hu1 = np.random.normal(7.5, 1.0)
                    sift = np.random.normal(70, 15)
                elif label == "Paisley":
                    # Tear-drop curves, highly unique hu moments, medium-high circularity
                    area = np.random.normal(7500, 1800)
                    perimeter = np.random.normal(450, 90)
                    circularity = np.random.uniform(0.6, 0.75)
                    texture = np.random.normal(10.0, 1.5)
                    hu0 = np.random.normal(2.8, 0.3)
                    hu1 = np.random.normal(6.5, 0.7)
                    sift = np.random.normal(45, 8)
                
                # Make sure values are physical
                circularity = max(0.01, min(1.0, circularity))
                sift = max(0, int(sift))
                
                X.append([area, perimeter, circularity, texture, hu0, hu1, sift])
                y.append(label_idx)
                
        X = np.array(X)
        y = np.array(y)
        
        # Scaler
        cls._scaler = StandardScaler()
        X_scaled = cls._scaler.fit_transform(X)
        
        # Train Random Forest
        cls._rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        cls._rf_model.fit(X_scaled, y)
        
        # Train SVM
        cls._svm_model = SVC(probability=True, random_state=42)
        cls._svm_model.fit(X_scaled, y)
        
        # Train Gradient Boosting (corresponds to XGBoost in sklearn ecosystem)
        cls._gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        cls._gb_model.fit(X_scaled, y)
        
        logger.info("Motif classification models initialized successfully.")

    @classmethod
    def classify_motif(cls, features: dict, model_type: str = "Random Forest") -> dict:
        """
        Classifies a single motif based on its extracted features.
        model_type: "Random Forest", "SVM", "XGBoost"
        """
        if cls._rf_model is None or cls._svm_model is None or cls._gb_model is None:
            cls.initialize_models()
            
        # Extract features vector
        area = features.get("area", 0.0)
        perimeter = features.get("perimeter", 0.0)
        circularity = features.get("circularity", 0.0)
        texture_score = features.get("texture_score", 0.0)
        hu = features.get("hu_moments", [0.0]*7)
        hu0 = hu[0] if len(hu) > 0 else 0.0
        hu1 = hu[1] if len(hu) > 1 else 0.0
        sift = features.get("sift_keypoints", 0)
        
        feat_vector = np.array([[area, perimeter, circularity, texture_score, hu0, hu1, sift]])
        feat_vector_scaled = cls._scaler.transform(feat_vector)
        
        # Select model
        if model_type == "SVM":
            model = cls._svm_model
        elif model_type == "XGBoost":
            model = cls._gb_model
        else:
            model = cls._rf_model
            
        probs = model.predict_proba(feat_vector_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        return {
            "predicted_class": cls.CLASSES[pred_idx],
            "confidence_score": float(confidence),
            "probabilities": {cls.CLASSES[i]: float(probs[i]) for i in range(len(cls.CLASSES))}
        }
