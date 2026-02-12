import cv2
import numpy as np
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

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
        score_threshold: float = 0.5
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
        polygon: List[float],
        scale_x: float,
        scale_y: float
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
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return []

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 1:
            return []

        coords = largest_contour.reshape(-1, 2).flatten().tolist()

        return coords


class ArcadeMetricsCalculator:
    """Calculate IoU and Dice metrics from ARCADE-formatted data."""

    @staticmethod
    def normalize_segmentation(segmentation) -> List[float]:
        """
        Normalize segmentation to flat list format.

        ARCADE segmentations can be:
        - Flat: [x1, y1, x2, y2, ...]
        - Nested: [[x1, y1, x2, y2, ...]]

        Args:
            segmentation: Segmentation in either format

        Returns:
            Flat list of coordinates
        """
        if not segmentation:
            return []

        if isinstance(segmentation[0], list):
            return segmentation[0]

        return segmentation

    @staticmethod
    def polygon_to_mask(
        segmentation,
        height: int,
        width: int
    ) -> np.ndarray:
        """
        Convert ARCADE polygon to binary mask.

        Args:
            segmentation: Polygon coordinates (flat or nested list)
            height: Image height
            width: Image width

        Returns:
            Binary mask as boolean numpy array
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        flat_seg = ArcadeMetricsCalculator.normalize_segmentation(segmentation)

        if not flat_seg or len(flat_seg) < 6:
            return mask.astype(bool)

        try:
            coords = np.array(flat_seg, dtype=np.float32).reshape(-1, 2)
            coords = coords.astype(np.int32)

            coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)

            cv2.fillPoly(mask, [coords], 1)
        except (ValueError, IndexError):
            pass

        return mask.astype(bool)

    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Intersection over Union for binary masks.

        Args:
            mask1, mask2: Binary masks as boolean numpy arrays

        Returns:
            IoU score in range [0, 1]
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return float(intersection / union)

    @staticmethod
    def calculate_dice(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Calculate Dice coefficient for binary masks.

        Args:
            mask1, mask2: Binary masks as boolean numpy arrays

        Returns:
            Dice score in range [0, 1]
        """
        intersection = np.logical_and(mask1, mask2).sum()
        total = mask1.sum() + mask2.sum()

        if total == 0:
            return 0.0

        return float((2.0 * intersection) / total)

    @staticmethod
    def match_predictions_to_ground_truth(
        pred_masks: List[np.ndarray],
        gt_masks: List[np.ndarray],
        iou_threshold: float = 0.5
    ) -> List[Tuple[int, int, float]]:
        """
        Match predictions to ground truth using Hungarian matching based on IoU.

        Args:
            pred_masks: List of predicted binary masks
            gt_masks: List of ground truth binary masks
            iou_threshold: Minimum IoU to consider a valid match

        Returns:
            List of (pred_idx, gt_idx, iou) tuples for matched pairs
        """
        if len(pred_masks) == 0 or len(gt_masks) == 0:
            return []

        iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))

        for i, pred_mask in enumerate(pred_masks):
            for j, gt_mask in enumerate(gt_masks):
                iou_matrix[i, j] = ArcadeMetricsCalculator.calculate_iou(
                    pred_mask, gt_mask
                )

        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)

        matches = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou >= iou_threshold:
                matches.append((pred_idx, gt_idx, iou))

        return matches

    def calculate_metrics_for_image(
        self,
        pred_annotations: List[Dict],
        gt_annotations: List[Dict],
        image_height: int,
        image_width: int,
        iou_threshold: float = 0.5
    ) -> Dict:
        """
        Calculate IoU and Dice metrics for a single image.

        All coordinates should be in the SAME space (original image space).

        Args:
            pred_annotations: List of predicted ARCADE annotations
            gt_annotations: List of ground truth ARCADE annotations
            image_height: Image height in pixels (original space)
            image_width: Image width in pixels (original space)
            iou_threshold: Minimum IoU for matching

        Returns:
            Dictionary with metrics
        """
        pred_masks = [
            self.polygon_to_mask(ann["segmentation"], image_height, image_width)
            for ann in pred_annotations
        ]

        gt_masks = [
            self.polygon_to_mask(ann["segmentation"], image_height, image_width)
            for ann in gt_annotations
        ]

        matches = self.match_predictions_to_ground_truth(
            pred_masks, gt_masks, iou_threshold
        )

        ious = []
        dices = []

        for pred_idx, gt_idx, iou in matches:
            pred_mask = pred_masks[pred_idx]
            gt_mask = gt_masks[gt_idx]

            dice = self.calculate_dice(pred_mask, gt_mask)

            ious.append(iou)
            dices.append(dice)

        return {
            "ious": ious,
            "dices": dices,
            "num_matches": len(matches),
            "num_predictions": len(pred_annotations),
            "num_ground_truth": len(gt_annotations)
        }