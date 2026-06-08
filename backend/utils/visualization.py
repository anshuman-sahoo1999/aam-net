import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class VisualizationUtil:
    @staticmethod
    def fig_to_base64(fig) -> str:
        """Converts a Matplotlib figure to a base64 encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    @staticmethod
    def draw_contour_overlay(img: np.ndarray, motifs: list) -> np.ndarray:
        """Draws contours and IDs over the original image."""
        overlay = img.copy()
        for motif in motifs:
            cv2.drawContours(overlay, [motif["contour"]], -1, (0, 255, 0), 2)
            # Label
            x, y, w, h = motif["bounding_box"]
            cv2.putText(
                overlay, f"ID: {motif['id']}", (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
        return overlay

    @staticmethod
    def draw_orientation_vectors(img: np.ndarray, angle_maps: list) -> np.ndarray:
        """Draws major (red) and minor (blue) orientation axes for each motif."""
        vector_img = img.copy()
        for m in angle_maps:
            cx, cy = int(m["x_coordinate"]), int(m["y_coordinate"])
            angle_rad = np.radians(m["major_axis_angle"])
            minor_rad = np.radians(m["minor_axis_angle"])
            
            # Use eigenvalues to scale vector lengths, set a reasonable range [20, 100]
            val = m.get("eigenvalues", [100.0, 50.0])
            len_major = int(np.clip(np.sqrt(val[0]) * 1.5, 20, 80))
            len_minor = int(np.clip(np.sqrt(val[1]) * 1.5, 10, 40))
            
            # Major vector end points
            x1 = int(cx + len_major * np.cos(angle_rad))
            y1 = int(cy + len_major * np.sin(angle_rad))
            x2 = int(cx - len_major * np.cos(angle_rad))
            y2 = int(cy - len_major * np.sin(angle_rad))
            
            # Minor vector end points
            mx1 = int(cx + len_minor * np.cos(minor_rad))
            my1 = int(cy + len_minor * np.sin(minor_rad))
            mx2 = int(cx - len_minor * np.cos(minor_rad))
            my2 = int(cy - len_minor * np.sin(minor_rad))
            
            # Draw lines and center circle
            cv2.line(vector_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for Major axis
            cv2.line(vector_img, (mx1, my1), (mx2, my2), (255, 0, 0), 1)  # Blue for Minor axis
            cv2.circle(vector_img, (cx, cy), 3, (0, 255, 255), -1)  # Yellow center
            
            # Label
            cv2.putText(
                vector_img, f"{int(m['angle_degree'])}deg", (cx + 5, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )
            
        return vector_img

    @staticmethod
    def generate_angle_heatmap(img: np.ndarray, motifs: list, angle_maps: list) -> np.ndarray:
        """Fills contours with a color mapped to their orientation angle."""
        h, w = img.shape[:2]
        heatmap = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Base image semi-transparent background
        background = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2
        
        # Color scale helper (using HSV to cycle colors through 180 degrees)
        # H in OpenCV is 0-180, so we can map angle_degree [0, 180] directly to H
        for motif, angle_info in zip(motifs, angle_maps):
            contour = motif["contour"]
            angle = angle_info["angle_degree"]
            
            # H: angle, S: 255, V: 255
            hsv_color = np.array([[[int(angle), 255, 255]]], dtype=np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
            
            cv2.drawContours(heatmap, [contour], -1, color, -1)
            cv2.drawContours(heatmap, [contour], -1, (255, 255, 255), 1)
            
        # Blend heatmap and original grayscale background
        mask = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY) > 0
        blended = background.copy()
        blended[mask] = cv2.addWeighted(background, 0.3, heatmap, 0.7, 0)[mask]
        
        return blended

    @staticmethod
    def generate_feature_distribution(angle_maps: list, features: list) -> str:
        """Generates a matplotlib scatter/distribution plot of orientation vs circularity/area."""
        if not angle_maps or not features:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No Data to Plot", ha='center', va='center')
            return VisualizationUtil.fig_to_base64(fig)
            
        angles = [am["angle_degree"] for am in angle_maps]
        circularities = [f["circularity"] for f in features]
        areas = [f["area"] for f in features]
        ids = [am["motif_id"] for am in angle_maps]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot 1: Angle vs Circularity
        scatter1 = ax1.scatter(angles, circularities, c=angles, cmap='hsv', s=100, edgecolors='black', alpha=0.85)
        ax1.set_title("Circularity vs. Orientation Angle", fontsize=11, fontweight='bold', pad=10)
        ax1.set_xlabel("Orientation Angle (Degrees)")
        ax1.set_ylabel("Circularity")
        ax1.set_xlim(-10, 190)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Label points
        for i, txt in enumerate(ids):
            ax1.annotate(f"#{txt}", (angles[i], circularities[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
            
        # Plot 2: Histogram of orientation angles
        n, bins, patches = ax2.hist(angles, bins=12, range=(0, 180), edgecolor='black', alpha=0.7)
        # Apply color map to histogram bars based on center angle
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        col = plt.cm.hsv(bin_centers / 180.0)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', c)
            
        ax2.set_title("Orientation Angle Frequency", fontsize=11, fontweight='bold', pad=10)
        ax2.set_xlabel("Angle (Degrees)")
        ax2.set_ylabel("Count")
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return VisualizationUtil.fig_to_base64(fig)
