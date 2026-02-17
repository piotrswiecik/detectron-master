import uuid

import cv2
import numpy as np

from coro_dt.data.converter import DetectronToArcadeConverter


def format_detectron(instances, image_path: str) -> dict:
    """Extract raw Detectron2 predictions as a JSON-serializable dict."""
    instances = instances.to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().tolist()
    classes = instances.pred_classes.numpy().tolist()
    scores = instances.scores.numpy().tolist()

    return {
        "image": image_path,
        "num_instances": len(instances),
        "instances": [
            {"bbox": bbox, "category_id": cat, "score": score}
            for bbox, cat, score in zip(boxes, classes, scores)
        ],
    }


def format_arcade(instances, image_id: int, height: int, width: int) -> dict:
    """Convert Detectron2 predictions to ARCADE annotation format."""
    converter = DetectronToArcadeConverter(category_id_reverse_map={0: 0})
    annotations = converter.convert_instances(
        instances=instances.to("cpu"),
        image_id=image_id,
        original_height=height,
        original_width=width,
        transformed_height=height,
        transformed_width=width,
        score_threshold=0.0,
    )
    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def format_labelstudio(instances, orig_height: int, orig_width: int) -> dict:
    """Convert Detectron2 predictions to Label Studio pre-annotation format."""
    try:
        from label_studio_converter import brush
    except ImportError:
        raise ImportError(
            "label-studio-converter is required for the 'labelstudio' format. "
            "Install it with: pip install -e '.[labelstudio]'"
        )

    instances = instances.to("cpu")
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()

    results = []
    for mask_bool, score in zip(masks, scores):
        if mask_bool.shape != (orig_height, orig_width):
            mask_uint8 = mask_bool.astype(np.uint8)
            mask_uint8 = cv2.resize(
                mask_uint8, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST
            )
            mask_bool = mask_uint8.astype(bool)

        if not mask_bool.any():
            continue

        m = mask_bool.astype(np.uint8) * 255
        rle = brush.mask2rle(m)

        results.append({
            "id": str(uuid.uuid4())[:8],
            "from_name": "brush_labels",
            "to_name": "image",
            "type": "brushlabels",
            "original_width": orig_width,
            "original_height": orig_height,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": rle,
                "brushlabels": ["vessel"],
            },
            "score": float(score),
        })

    overall = float(np.mean([r["score"] for r in results])) if results else 0.0

    return {
        "result": results,
        "score": overall,
    }
