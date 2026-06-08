import cv2
import numpy as np

# Try importing skimage.feature, fallback to manual numpy implementation if unavailable
try:
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class FeatureExtractorService:
    @staticmethod
    def calculate_hu_moments(contour) -> list:
        """Calculates log-transformed Hu Moments for shape scale/rotation invariance."""
        moments = cv2.moments(contour)
        hu = cv2.HuMoments(moments).flatten()
        # Log-transform Hu moments to scale them properly
        log_hu = []
        for h in hu:
            if h != 0:
                log_hu.append(-1.0 * np.sign(h) * np.log10(np.abs(h)))
            else:
                log_hu.append(0.0)
        return log_hu

    @staticmethod
    def extract_hog_features(img: np.ndarray) -> np.ndarray:
        """Extracts standard HOG features after resizing the motif region to 64x64."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (64, 64))
        
        # Instantiate HOG descriptor
        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )
        try:
            features = hog.compute(resized)
            return features.flatten().tolist()[:100]  # Cap at 100 features for response sizing
        except Exception:
            return [0.0] * 100

    @staticmethod
    def extract_sift_features(img: np.ndarray) -> int:
        """Extracts SIFT keypoints and returns the count of keypoints as a complexity measure."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return len(keypoints)

    @staticmethod
    def calculate_glcm_texture(img: np.ndarray) -> dict:
        """Calculates GLCM texture properties (Contrast, Correlation, Energy, Homogeneity)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # Crop black margins
        non_zero = np.argwhere(gray > 0)
        if non_zero.size > 0:
            top_left = non_zero.min(axis=0)
            bottom_right = non_zero.max(axis=0)
            gray = gray[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
            
        if gray.size == 0 or gray.max() == 0:
            return {"contrast": 0.0, "correlation": 0.0, "energy": 0.0, "homogeneity": 0.0}

        # Downscale gray levels to 16 to reduce computations
        quantized = (gray // 16).astype(np.uint8)
        
        if HAS_SKIMAGE:
            try:
                glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
                contrast = float(graycoprops(glcm, 'contrast')[0, 0])
                correlation = float(graycoprops(glcm, 'correlation')[0, 0])
                energy = float(graycoprops(glcm, 'energy')[0, 0])
                homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
                return {
                    "contrast": contrast,
                    "correlation": correlation,
                    "energy": energy,
                    "homogeneity": homogeneity
                }
            except Exception:
                pass
                
        # NumPy Fallback implementation of GLCM
        levels = 16
        glcm = np.zeros((levels, levels), dtype=np.float32)
        h, w = quantized.shape
        count = 0
        for i in range(h):
            for j in range(w - 1):
                glcm[quantized[i, j], quantized[i, j+1]] += 1
                glcm[quantized[i, j+1], quantized[i, j]] += 1
                count += 2
        
        if count > 0:
            glcm /= count
            
        contrast = 0.0
        correlation = 0.0
        energy = 0.0
        homogeneity = 0.0
        
        # Calculate properties
        mean_i = 0.0
        mean_j = 0.0
        for i in range(levels):
            for j in range(levels):
                val = glcm[i, j]
                contrast += val * ((i - j) ** 2)
                energy += val ** 2
                homogeneity += val / (1.0 + (i - j) ** 2)
                mean_i += i * val
                mean_j += j * val
                
        # Std dev for correlation
        var_i = 0.0
        var_j = 0.0
        for i in range(levels):
            for j in range(levels):
                val = glcm[i, j]
                var_i += ((i - mean_i) ** 2) * val
                var_j += ((j - mean_j) ** 2) * val
                
        std_i = np.sqrt(var_i)
        std_j = np.sqrt(var_j)
        
        if std_i > 0 and std_j > 0:
            for i in range(levels):
                for j in range(levels):
                    correlation += glcm[i, j] * (i - mean_i) * (j - mean_j) / (std_i * std_j)
                    
        return {
            "contrast": float(contrast),
            "correlation": float(correlation),
            "energy": float(energy),
            "homogeneity": float(homogeneity)
        }

    @classmethod
    def extract_all_features(cls, motif: dict) -> dict:
        """Runs all extraction logic and returns results for a motif."""
        img = motif["segmented"]
        contour = motif["contour"]
        
        glcm_feats = cls.calculate_glcm_texture(img)
        hu_moments = cls.calculate_hu_moments(contour)
        hog_features = cls.extract_hog_features(img)
        sift_kp_count = cls.extract_sift_features(img)
        
        # Combine texture elements into a single texture score metric
        texture_score = glcm_feats["energy"] * 10.0 + glcm_feats["homogeneity"] * 5.0 - glcm_feats["contrast"] * 0.1
        
        return {
            "motif_id": motif["id"],
            "area": float(motif["area"]),
            "perimeter": float(motif["perimeter"]),
            "circularity": float(motif["circularity"]),
            "hu_moments": hu_moments,
            "hog": hog_features,
            "sift_keypoints": sift_kp_count,
            "texture": glcm_feats,
            "texture_score": float(texture_score),
            "dominant_orientation": 0.0  # Computed in next phase
        }
