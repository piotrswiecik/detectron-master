# Detectron V2 tuning

## Installation

This model requires special workflow. Model must be installed directly from external repo
as a separate dependency.

```shell
# install normal dependencies
uv sync

# install detectron but when building source - reuse dependencies from step 1
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# install cli tools
uv pip install -e .

# optional: install label studio format support
uv pip install -e ".[labelstudio]"

# optional: locally install jupyter
uv pip install ipykernel
```

## Training commands

```shell
train-multi --data-root /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax --epochs 1 --batch-size 2 --params-file params.json

train-multi --data-root /home/ives/piotr/arcade/syntax --epochs 1 --batch-size 2 --params-file params.json
```

## Sampling commands

```shell
sample-multi /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json

# default format (detectron) — bounding boxes, class IDs, scores
sample-binary /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json

# arcade format — polygon segmentations in ARCADE annotation space
sample-binary /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json --prediction-format arcade
sample-binary /Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/images/1.png --use-cpu --threshold 0.1 --params-file params.json --prediction-format arcade

# label studio format — RLE-encoded masks for Label Studio pre-annotations (requires optional dependency, see below)
sample-binary /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json --prediction-format labelstudio

# persist prediction JSON to a file in the current directory
sample-binary /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json --prediction-format detectron --store
```

## Binary finetuning

```shell
train-binary --data-root /home/ives/piotr/arcade/syntax --weights /path/to/multi-class/output/model_final.pth --epochs 5
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

Shape of Label Studio pre-annotation (brushlabels with RLE-encoded masks).

```json
{
    "result": [
        {
            "id": "abc12345",
            "from_name": "brush_labels",
            "to_name": "image",
            "type": "brushlabels",
            "original_width": 512,
            "original_height": 512,
            "image_rotation": 0,
            "value": {
                "format": "rle",
                "rle": [0, 10, 1, 5, "..."],
                "brushlabels": ["vessel"]
            },
            "score": 0.95
        }
    ],
    "score": 0.85
}
```
