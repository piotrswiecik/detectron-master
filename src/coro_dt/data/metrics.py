import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


from typing import Dict, List, Tuple


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
    def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
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
        iou_threshold: float = 0.5,
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
        iou_threshold: float = 0.5,
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
            "num_ground_truth": len(gt_annotations),
        }
