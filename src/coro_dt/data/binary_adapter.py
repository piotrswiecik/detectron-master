"""
ARCADE syntax dataset binary adapter. Converts input dataset annotations into single vessel class format.

Each original annotation keeps its own bounding box and segmentation — only the category_id
is remapped to 0 ("vessel"). This preserves per-instance detection targets that are
appropriately sized for the anchor/proposal mechanism.

The expected output format for each iterator call is a dictionary with the following structure:
{
    "file_name": "/path/to/image.png",
    "height": 512,
    "width": 512,
    "image_id": 922,
    "annotations": [
        {
            "bbox": [x1, y1, x2, y2],
            "bbox_mode": 0,  # XYXY_ABS
            "category_id": 0,  # Always 0 (vessel)
            "segmentation": [[x1,y1,..., xn,yn]],
        },
        ...
    ]
}
"""

import os
from collections import defaultdict
from collections.abc import Iterator


class BinaryAdapter:
    def __init__(self, arcade: dict, image_root: str):
        self.images = [img for img in arcade.get("images", [])]
        self._raw_anns = [ann for ann in arcade.get("annotations", [])]
        self.image_root = image_root

        self._grouped_anns = defaultdict(list)
        for ann in self._raw_anns:
            self._grouped_anns[ann["image_id"]].append(ann)

        self._box_mode = 0  # XYXY_ABS

        self.id_map = {0: 0}
        self.class_names = ["vessel"]

    @staticmethod
    def _calculate_xyxyabs_bbox(segmentation: list) -> list[float]:
        """Calculate bounding box in XYXY_ABS format from segmentation."""
        if not segmentation:
            return [0.0, 0.0, 0.0, 0.0]

        if isinstance(segmentation[0], list):
            flat_coords = [c for poly in segmentation for c in poly]
        else:
            flat_coords = segmentation

        if not flat_coords:
            return [0.0, 0.0, 0.0, 0.0]

        xs = flat_coords[0::2]
        ys = flat_coords[1::2]
        return [min(xs), min(ys), max(xs), max(ys)]

    def _convert_annotation(self, ann: dict) -> dict:
        raw_seg = ann["segmentation"]
        bbox = self._calculate_xyxyabs_bbox(raw_seg)

        final_seg = (
            [raw_seg] if raw_seg and not isinstance(raw_seg[0], list) else raw_seg
        )

        return {
            "bbox": bbox,
            "bbox_mode": self._box_mode,
            "category_id": 0,
            "segmentation": final_seg,
        }

    def __iter__(self) -> Iterator[dict]:
        for img in self.images:
            related_anns = self._grouped_anns.get(img["id"], [])
            converted_anns = [self._convert_annotation(ann) for ann in related_anns]

            abs_path = os.path.abspath(os.path.join(self.image_root, img["file_name"]))

            yield {
                "file_name": abs_path,
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
                "annotations": converted_anns,
            }

    def as_list(self) -> list[dict]:
        return list(self)
