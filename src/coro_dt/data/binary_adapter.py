"""
ARCADE syntax dataset binary adapter. Converts input dataset annotations into single vessel class format.

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
            "segmentation": [[x1,y1,...], [x1,y1,...]],  # 1+ polygons
        }
    ]
}
"""

from collections import defaultdict
from collections.abc import Iterator
import os
from shapely import geometry
from shapely.ops import unary_union
from shapely.validation import make_valid


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
    def _segmentation_to_polygon(segmentation) -> geometry.Polygon | None:
        """
        Convert ARCADE segmentation to Shapely Polygon.

        Args:
            segmentation: Flat list [x1, y1, x2, y2, ...] or nested [[x1, y1, ...]]

        Returns:
            Shapely Polygon or None if invalid
        """
        if segmentation and isinstance(segmentation[0], list):
            coords = segmentation[0]
        else:
            coords = segmentation

        if not coords or len(coords) < 6:
            return None

        points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]

        try:
            poly = geometry.Polygon(points)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or not isinstance(poly, (geometry.Polygon, geometry.MultiPolygon)):
                return None
            return poly
        except Exception:
            return None

    @staticmethod
    def _polygon_to_segmentation(poly: geometry.Polygon) -> list[float]:
        """
        Convert Shapely Polygon back to ARCADE flat coordinate list.

        Args:
            poly: Shapely Polygon

        Returns:
            Flat list [x1, y1, x2, y2, ...]
        """
        coords = list(poly.exterior.coords)[:-1]
        return [c for point in coords for c in point]

    @staticmethod
    def _calculate_xyxyabs_bbox(segmentation: list) -> list[float]:
        """Calculate bounding box in XYXY_ABS format from segmentation."""
        if not segmentation:
            return [0.0, 0.0, 0.0, 0.0]

        xs = segmentation[0::2]
        ys = segmentation[1::2]

        if not xs or not ys:
            return [0.0, 0.0, 0.0, 0.0]

        return [min(xs), min(ys), max(xs), max(ys)]

    def _merge_annotations(self, annotations: list[dict]) -> dict | None:
        """
        Merge multiple annotations into a single vessel annotation.

        Args:
            annotations: List of ARCADE annotation dicts for one image

        Returns:
            Single merged annotation dict, or None if no valid polygons
        """
        polygons = []

        for ann in annotations:
            poly = self._segmentation_to_polygon(ann.get("segmentation", []))
            if poly is not None:
                if isinstance(poly, geometry.MultiPolygon):
                    polygons.extend(poly.geoms)
                else:
                    polygons.append(poly)

        if not polygons:
            return None

        try:
            merged = unary_union(polygons)
        except Exception:
            return None

        if merged.is_empty:
            return None

        if isinstance(merged, geometry.MultiPolygon):
            segmentations = []
            for geom in merged.geoms:
                if isinstance(geom, geometry.Polygon) and not geom.is_empty:
                    seg = self._polygon_to_segmentation(geom)
                    if len(seg) >= 6:
                        segmentations.append(seg)

            if not segmentations:
                return None

            all_coords = [c for seg in segmentations for c in seg]
            bbox = self._calculate_xyxyabs_bbox(all_coords)

            return {
                "bbox": bbox,
                "bbox_mode": self._box_mode,
                "category_id": 0,
                "segmentation": segmentations,
            }
        elif isinstance(merged, geometry.Polygon):
            segmentation = self._polygon_to_segmentation(merged)

            if len(segmentation) < 6:
                return None

            bbox = self._calculate_xyxyabs_bbox(segmentation)

            return {
                "bbox": bbox,
                "bbox_mode": self._box_mode,
                "category_id": 0,
                "segmentation": [segmentation],
            }
        else:
            return None

    def __iter__(self) -> Iterator[dict]:
        for img in self.images:
            related_anns = self._grouped_anns.get(img["id"], [])

            abs_path = os.path.abspath(os.path.join(self.image_root, img["file_name"]))

            result = {
                "file_name": abs_path,
                "height": img["height"],
                "width": img["width"],
                "image_id": img["id"],
                "annotations": [],
            }

            if related_anns:
                merged = self._merge_annotations(related_anns)
                if merged is not None:
                    result["annotations"] = [merged]

            yield result

    def as_list(self) -> list[dict]:
        return list(self)
