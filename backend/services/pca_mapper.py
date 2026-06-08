import numpy as np

class PCAMapperService:
    @staticmethod
    def calculate_orientation(contour) -> dict:
        """
        Applies PCA on the 2D contour points to calculate the principal orientation angle.
        Returns:
            - angle_degree: Principal angle of orientation (major axis) in [-90, 90] or [0, 180] degrees.
            - major_axis_angle: Angle of the major eigenvector.
            - minor_axis_angle: Angle of the minor eigenvector.
            - eigenvalues: Variance along major and minor axes.
            - center: (x, y) coordinates of the centroid.
        """
        # Reshape contour points to (N, 2)
        pts = contour.reshape(-1, 2).astype(np.float64)
        
        if len(pts) < 3:
            # Fallback if there are too few points
            return {
                "angle_degree": 0.0,
                "major_axis_angle": 0.0,
                "minor_axis_angle": 90.0,
                "eigenvalues": [0.0, 0.0],
                "center": (0.0, 0.0)
            }
            
        # Centroid
        mean = np.mean(pts, axis=0)
        center_x, center_y = mean[0], mean[1]
        
        # Center the data points
        pts_centered = pts - mean
        
        # Covariance matrix
        cov = np.cov(pts_centered, rowvar=False)
        
        # Eigen decomposition
        evals, evecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(evals)[::-1]
        evals = evals[sort_indices]
        evecs = evecs[:, sort_indices]
        
        # Major eigenvector
        v1 = evecs[:, 0]  # First principal component
        v2 = evecs[:, 1]  # Second principal component
        
        # Calculate angles in degrees
        # Note: In image coordinates, y increases downwards, so we calculate angle relative to horizontal x-axis.
        major_axis_angle = np.degrees(np.arctan2(v1[1], v1[0]))
        minor_axis_angle = np.degrees(np.arctan2(v2[1], v2[0]))
        
        # Normalize principal angle to range [0, 180] or [-90, 90]
        # In textile informatics, orientation is typically represented modulo 180 degrees.
        angle_degree = major_axis_angle % 180.0
        
        return {
            "angle_degree": float(angle_degree),
            "major_axis_angle": float(major_axis_angle),
            "minor_axis_angle": float(minor_axis_angle),
            "eigenvalues": [float(evals[0]), float(evals[1])],
            "center": (float(center_x), float(center_y))
        }

    @classmethod
    def map_motifs_angles(cls, motifs: list) -> list:
        """
        Maps angles for all motifs and formats the output.
        """
        angle_maps = []
        for motif in motifs:
            pca_data = cls.calculate_orientation(motif["contour"])
            angle_maps.append({
                "motif_id": motif["id"],
                "angle_degree": pca_data["angle_degree"],
                "major_axis_angle": pca_data["major_axis_angle"],
                "minor_axis_angle": pca_data["minor_axis_angle"],
                "x_coordinate": pca_data["center"][0],
                "y_coordinate": pca_data["center"][1],
                "eigenvalues": pca_data["eigenvalues"]
            })
        return angle_maps
