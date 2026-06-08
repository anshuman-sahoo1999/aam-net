import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessingService:
    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """Converts raw image bytes to an OpenCV BGR image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
        return img

    @staticmethod
    def cv2_to_bytes(img: np.ndarray, format: str = ".png") -> bytes:
        """Converts an OpenCV image back to raw bytes."""
        success, encoded_img = cv2.imencode(format, img)
        if not success:
            raise ValueError("Could not encode image to bytes.")
        return encoded_img.tobytes()

    @staticmethod
    def preprocess_image(img: np.ndarray) -> dict:
        """
        Applies:
        1. RGB to Grayscale
        2. Histogram Equalization
        3. Gaussian Denoising
        4. Adaptive Thresholding
        Returns dictionary containing intermediate and final processed images as cv2 images.
        """
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Histogram Equalization (CLAHE is preferred for local contrast in textiles)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # 3. Gaussian Denoising
        denoised = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        # 4. Adaptive Thresholding
        # Using ADAPTIVE_THRESH_GAUSSIAN_C to capture detailed textile motifs
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return {
            "gray": gray,
            "equalized": equalized,
            "denoised": denoised,
            "thresh": thresh
        }

    @staticmethod
    def detect_motifs(thresh_img: np.ndarray, original_img: np.ndarray, min_area: float = 100.0, max_area: float = 100000.0) -> list:
        """
        Detects repeating motifs by finding contours, filtering by area size, and extracting segmented regions.
        Returns a list of dictionaries with motif information, including coordinates, contours, and cutouts.
        """
        # Find contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motifs = []
        motif_id = 1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter contours by size to ignore tiny noise or excessively large background elements
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Extract segmented region (bounding box)
                cutout = original_img[y:y+h, x:x+w]
                # Create a mask for exact contour region
                mask = np.zeros(thresh_img.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mask_cutout = mask[y:y+h, x:x+w]
                
                # Apply mask to cutout to keep only the motif on black background
                segmented = cv2.bitwise_and(cutout, cutout, mask=mask_cutout)
                
                motifs.append({
                    "id": motif_id,
                    "contour": contour,
                    "bounding_box": (x, y, w, h),
                    "area": area,
                    "perimeter": perimeter,
                    "circularity": circularity,
                    "segmented": segmented,
                    "mask_cutout": mask_cutout,
                    "centroid": (int(x + w / 2), int(y + h / 2))
                })
                motif_id += 1
                
        return motifs
