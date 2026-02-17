import glob
import json
import os
from pathlib import Path

import cv2
import typer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from coro_dt.config import ParamsConfig
from coro_dt.data.formatters import format_arcade, format_detectron, format_labelstudio

app = typer.Typer()


def find_and_select_model(models_dir: str) -> str:
    """Recursively find all .pth files in a directory and let the user pick one."""
    pth_files = sorted(glob.glob(os.path.join(models_dir, "**", "*.pth"), recursive=True))
    if not pth_files:
        typer.echo(f"No .pth files found in {models_dir}")
        raise typer.Exit(code=1)

    typer.echo(f"\nFound {len(pth_files)} model(s) in {models_dir}:\n")
    for i, path in enumerate(pth_files, start=1):
        rel = os.path.relpath(path, models_dir)
        typer.echo(f"  [{i}] {rel}")

    typer.echo()
    choice = typer.prompt("Select a model", type=int)
    if choice < 1 or choice > len(pth_files):
        typer.echo("Invalid selection.")
        raise typer.Exit(code=1)

    return pth_files[choice - 1]


@app.command()
def infer(
    image_path: str,
    models_dir: str = typer.Option(
        "trained_models", help="Directory to search for .pth model files"
    ),
    threshold: float = typer.Option(
        0.5, help="Minimum score threshold to display a prediction"
    ),
    params_file: str = typer.Option(
        ..., help="Path to the JSON parameters file used during training (required to match model architecture)"
    ),
    use_cpu: bool = typer.Option(False, help="Force inference on CPU"),
    prediction_format: str = typer.Option(
        "detectron", help="Output format: detectron, arcade, or labelstudio"
    ),
    store: bool = typer.Option(False, help="Write prediction JSON to a file in CWD"),
):
    """
    Run inference on a single image using a trained binary vessel detection model.
    """
    NUM_CLASSES = 1

    with open(params_file, "r") as f:
        config = ParamsConfig(**json.load(f))

    weights_path = find_and_select_model(models_dir)
    typer.echo(f"Selected: {weights_path}")
    typer.echo(f"Using backbone: {config.backbone.value}")

    if not os.path.exists(image_path):
        typer.echo(f"Error: Image not found at {image_path}")
        raise typer.Exit(code=1)

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(config.backbone.value)
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = config.anchor_sizes
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = config.anchor_ratios
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.roi_batch_size
    cfg.MODEL.BACKBONE.FREEZE_AT = config.freeze_at

    if use_cpu:
        cfg.MODEL.DEVICE = "cpu"

    print(f"Loading model from {weights_path}...")
    predictor = DefaultPredictor(cfg)

    print(f"Processing {image_path}...")
    im = cv2.imread(image_path)
    if im is None:
        typer.echo("Failed to read image.")
        raise typer.Exit(code=1)

    outputs = predictor(im)

    instances = outputs["instances"]
    print(f"Found {len(instances)} detected instances.")

    h, w = im.shape[:2]

    if prediction_format == "detectron":
        result = format_detectron(instances, image_path)
    elif prediction_format == "arcade":
        result = format_arcade(instances, image_id=0, height=h, width=w)
    elif prediction_format == "labelstudio":
        result = format_labelstudio(instances, orig_height=h, orig_width=w)
    else:
        typer.echo(f"Unknown format: {prediction_format}. Use detectron, arcade, or labelstudio.")
        raise typer.Exit(code=1)

    json_output = json.dumps(result, indent=2)
    typer.echo(json_output)

    if store:
        stem = Path(image_path).stem
        out_path = Path.cwd() / f"{stem}_{prediction_format}.json"
        out_path.write_text(json_output)
        typer.echo(f"Saved to {out_path}")


if __name__ == "__main__":
    app()
