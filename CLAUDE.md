# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coronary vessel instance segmentation using Detectron2 (Mask R-CNN) on the ARCADE syntax dataset. The project supports two training modes: **multi-class** (25 vessel segment categories) and **binary** (single "vessel" class). Binary mode can optionally use **PointRend** for sharper mask boundaries. Experiment tracking uses MLflow.

## Setup

```shell
uv sync
uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
uv pip install -e .
```

Detectron2 must be installed separately from source (not in pyproject.toml) because it requires `--no-build-isolation` to reuse torch/torchvision from the venv.

## Commands

### Training
```shell
train-multi --data-root /path/to/arcade/syntax --epochs 10 --batch-size 2 --params-file params.json
train-binary --data-root /path/to/arcade/syntax --epochs 10 --batch-size 2 --params-file params.json
train-binary --data-root /path/to/arcade/syntax --weights /path/to/model_final.pth --epochs 5  # finetune from multi-class
```

### Inference
```shell
sample-binary /path/to/image.png --use-cpu --threshold 0.1 --params-file params.json
sample-binary /path/to/image.png --params-file params.json --prediction-format arcade  # or: detectron, labelstudio
sample-multi /path/to/image.png --params-file params.json
```

### GUI
```shell
gui  # launches Streamlit app for interactive binary inference
```

### Testing & Linting
```shell
uv run pytest
uv run ruff check src/
uv run ruff format src/
```

## Architecture

### Training Pipeline (two parallel paths)

**Multi-class:** `cli/train_multi.py` -> `ArcadeOrchestrator` -> `ArcadeTrainer` (extends `DefaultTrainer`)
- Data adapter: `data/adapter.py` `Adapter` — converts ARCADE JSON to Detectron2 format, remaps category IDs to contiguous 0..N
- Hooks: `training/multi/hooks.py` — `EvalHook` (validation loss + ARCADE IoU/Dice metrics), `MLFlowHook` (logs to MLflow)

**Binary:** `cli/train_binary.py` -> `BinaryOrchestrator` -> `BinaryTrainer` (extends `DefaultTrainer`)
- Data adapter: `data/binary_adapter.py` `BinaryAdapter` — same as Adapter but remaps all categories to 0 ("vessel")
- Hooks: `training/binary/hooks.py` — `BinaryEvalHook`, `BinaryMLFlowHook`
- Supports `--weights` for finetuning from a multi-class checkpoint
- PointRend support: enabled via `"use_pointrend": true` in params.json; config applied in `config.py:apply_pointrend_overrides()`

**Shared components:**
- `training/multi/mappers.py` — `build_custom_mapper` (training augmentations) and `build_validation_mapper` (eval-only resize). Both binary and multi trainers use these mappers.
- `config.py` — `ParamsConfig` (Pydantic model for params.json: backbone, LR, anchors, PointRend flag), `Backbone` enum (R_50, R_101)
- `augmentation/frame.py` — `FrameAugmentationWrapper` (Detectron2 augmentation) wrapping `FrameTransform` (Albumentations). Simulates angiographic dark frames via crop/resize methods with Shapely-based polygon clipping.

### Inference Pipeline

`cli/sample_binary.py` and `cli/sample_multi.py` — Typer CLI apps that load a model via `DefaultPredictor` and output predictions in one of three formats.

### Output Formatters (`data/formatters.py`)
- `format_detectron` — raw bboxes, classes, scores
- `format_arcade` — polygon segmentations scaled back to original image space (via `DetectronToArcadeConverter`)
- `format_labelstudio` — RLE-encoded masks for Label Studio pre-annotations (requires `label-studio-converter`)

### Evaluation Metrics (`data/metrics.py`)
`ArcadeMetricsCalculator` — Hungarian matching of predicted vs. ground-truth polygons, computing per-instance IoU and Dice scores.

### GUI (`gui/app.py`)
Streamlit app for binary inference. `gui/run.py` is the entrypoint wrapper.

## Key Configuration

`params.json` is the single source of truth for model architecture params shared between training and inference. Fields defined in `ParamsConfig`:
- `backbone` — must match a `Backbone` enum value (R_50 or R_101 FPN config)
- `anchor_sizes` — must be exactly 5 groups for FPN levels
- `use_pointrend` — enables PointRend mask head (binary only)

Environment variables (loaded from `.env`): `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT`.

## Data Format

ARCADE syntax dataset expected layout:
```
data-root/
  train/
    images/
    annotations/train.json
  val/
    images/
    annotations/val.json
```

Annotations follow COCO-like format with polygon segmentations. Adapters convert ARCADE category IDs to contiguous Detectron2 IDs (multi) or collapse all to 0 (binary).
