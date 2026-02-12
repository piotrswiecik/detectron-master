import glob
import json
import os
import cv2
import typer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from coro_dt.config import ParamsConfig

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
        None, help="Path to a JSON parameters file (uses backbone setting from it)"
    ),
    use_cpu: bool = typer.Option(False, help="Force inference on CPU"),
):
    """
    Run inference on a single image using a trained Detectron2 model.
    """
    NUM_CLASSES = 25

    if params_file is not None:
        with open(params_file, "r") as f:
            config = ParamsConfig(**json.load(f))
    else:
        config = ParamsConfig()

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

    temp_metadata = MetadataCatalog.get("temp_inference")
    temp_metadata.set(thing_classes=[f"class_{i}" for i in range(NUM_CLASSES)])

    v = Visualizer(
        im[:, :, ::-1], metadata=temp_metadata, scale=1.0, instance_mode=ColorMode.IMAGE
    )

    out = v.draw_instance_predictions(instances.to("cpu"))

    result_image = out.get_image()[:, :, ::-1]

    cv2.namedWindow("Inference Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Inference Result", result_image)

    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
