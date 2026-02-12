# Detectron V2 tuning

## Installation

This model requires special workflow. Model must be installed directly from external repo
as a separate dependency.

```shell
# install normal dependencies
uv sync

# install detectron but when building source - reuse dependencies from step 1
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

## Data reference

Shape of ARCADE annotation.

```json
{
  "images": [
    {
      "id": 922,
      "width": 512,
      "height": 512,
      "file_name": "922.png",
      "license": 0,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 922,
      "category_id": 8,
      "segmentation": [
        382.0,
        350.75,
      ]
    }
  ]
}
```

Shape of Detectron annotation.

```json
{
    "file_name": "path",
    "height": 100,
    "width": 100,
    "image_id": 1,
    "annotations": [
        {
            "bbox": [x1, y1, width, height],
            "bbox_mode": 0, 
            "category_id": 1,
            "segmentation": [[x1, y1, ..., xn, yn], ... [x1, y1, ..., xn, yn]],
            "keypoints": [x1, y1, v1, ..., xn, yn, vn]
        }
}
```

Bbox modes: XYXY_ABS = 0, XYWH_ABS = 1, XYXY_REL = 2, XYWH_REL = 3, XYWHA_ABS = 4
