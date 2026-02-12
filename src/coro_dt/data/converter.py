import cv2
import numpy as np
from typing import List, Dict


class DetectronToArcadeConverter:
    """Convert Detectron2 predictions to ARCADE format with coordinate scaling."""

    def __init__(self, category_id_reverse_map: Dict[int, int]):
        """
        Args:
            category_id_reverse_map: Maps Detectron category_id --> ARCADE category_id
        """
        self.reverse_map = category_id_reverse_map
        self._annotation_counter = 0

    def reset_counter(self):
        """Reset annotation counter (call at start of each eval epoch)."""
        self._annotation_counter = 0

    def convert_instances(
        self,
        instances,
        image_id: int,
        original_height: int,
        original_width: int,
        transformed_height: int,
        transformed_width: int,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Convert Detectron2 Instances to ARCADE annotations, scaling coordinates
        from transformed space back to original image space.

        Args:
            instances: Detectron2 Instances object with pred_masks, scores, pred_classes
            image_id: Original ARCADE image ID
            original_height: Original image height (ARCADE space)
            original_width: Original image width (ARCADE space)
            transformed_height: Transformed image height (inference space)
            transformed_width: Transformed image width (inference space)
            score_threshold: Minimum confidence score to include prediction

        Returns:
            List of ARCADE annotation dicts with coordinates in ORIGINAL space
        """
        if len(instances) == 0:
            return []

        if not instances.has("pred_masks"):
            return []

        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        scale_x = original_width / transformed_width
        scale_y = original_height / transformed_height

        arcade_annotations = []

        for mask, score, pred_class in zip(masks, scores, pred_classes):
            if score < score_threshold:
                continue

            segmentation_transformed = self._mask_to_polygon(mask)

            if not segmentation_transformed:
                continue

            segmentation_original = self._scale_polygon(
                segmentation_transformed, scale_x, scale_y
            )

            self._annotation_counter += 1

            arcade_ann = {
                "id": self._annotation_counter,
                "image_id": image_id,
                "category_id": self.reverse_map.get(int(pred_class), int(pred_class)),
                "segmentation": segmentation_original,
                "score": float(score),
            }

            arcade_annotations.append(arcade_ann)

        return arcade_annotations

    @staticmethod
    def _scale_polygon(
        polygon: List[float], scale_x: float, scale_y: float
    ) -> List[float]:
        """
        Scale polygon coordinates by given factors.

        Args:
            polygon: Flat list [x1, y1, x2, y2, ...]
            scale_x: X scale factor
            scale_y: Y scale factor

        Returns:
            Scaled polygon coordinates
        """
        scaled = []
        for i in range(0, len(polygon), 2):
            scaled.append(polygon[i] * scale_x)
            scaled.append(polygon[i + 1] * scale_y)
        return scaled

    @staticmethod
    def _mask_to_polygon(binary_mask: np.ndarray) -> List[float]:
        """
        Convert binary mask to polygon coordinates in ARCADE format.

        Handles multiple mask dtypes:
        - bool: multiply by 255
        - float (0-1): threshold at 0.5, multiply by 255
        - uint8 (0-255): use directly

        Args:
            binary_mask: 2D array (bool, float, or uint8)

        Returns:
            Flattened list of polygon coordinates [x1, y1, x2, y2, ...]
        """
        if binary_mask.dtype == bool:
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
        elif binary_mask.dtype in [np.float32, np.float64]:
            mask_uint8 = ((binary_mask > 0.5) * 255).astype(np.uint8)
        elif binary_mask.max() <= 1:
            mask_uint8 = (binary_mask * 255).astype(np.uint8)
        else:
            mask_uint8 = binary_mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return []

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1:
            return []

        coords = largest_contour.reshape(-1, 2).flatten().tolist()

        return coords
