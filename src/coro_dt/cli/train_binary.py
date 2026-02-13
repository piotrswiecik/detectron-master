from datetime import datetime
import os
import json
import typer

from coro_dt.config import ParamsConfig
from coro_dt.training.binary.trainer import BinaryOrchestrator

app = typer.Typer()


@app.command()
def train(
    data_root: str = typer.Option(
        ..., help="Path to the root directory of ARCADE syntax dataset."
    ),
    epochs: int = typer.Option(10, help="Number of training epochs."),
    batch_size: int = typer.Option(2, help="Batch size for training."),
    params_file: str = typer.Option(
        ..., help="Path to the JSON parameters file for training configuration (required to define model architecture)."
    ),
    output_dir: str | None = typer.Option(
        None, help="Directory to save training outputs and checkpoints."
    ),
    weights: str | None = typer.Option(
        None, help="Path to a checkpoint file (e.g. model_final.pth from a previous training run). If not provided, starts from COCO pretrained weights."
    ),
):
    verify_data_root(data_root)
    typer.echo("Data root directory OK.")

    config = load_config_object(params_file)
    typer.echo(f"Loaded training configuration: {config}")

    output_dir = ensure_output_dir(output_dir)
    typer.echo(f"Training outputs will be saved to: {output_dir}")

    if weights is not None and not os.path.isfile(weights):
        raise typer.BadParameter(f"Weights file not found: {weights}")

    orchestrator = BinaryOrchestrator(data_root, output_dir, config, weights=weights)
    orchestrator.train(epochs, batch_size)


def ensure_output_dir(output_dir: str | None) -> str:
    """Set the output directory for training outputs. If None, use a default timestamped path."""
    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), "output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def verify_data_root(data_root: str):
    """Verify existence and proper structure of the data_root ARCADE syntax directory. Throws ValueError if checks fail."""
    if not os.path.exists(data_root):
        raise ValueError(f"The specified data_root path does not exist: {data_root}")

    if not os.path.isdir(data_root):
        raise ValueError(
            f"The specified data_root path is not a directory: {data_root}"
        )
    subdirs = os.listdir(data_root)

    if "train" not in subdirs or "val" not in subdirs:
        raise ValueError(
            f"The data_root directory must contain 'train' and 'val' subdirectories."
        )
    train_subdirs = os.listdir(os.path.join(data_root, "train"))
    val_subdirs = os.listdir(os.path.join(data_root, "val"))
    required_subdirs = {"images", "annotations"}

    if not required_subdirs.issubset(set(train_subdirs)):
        raise ValueError(
            f"The 'train' subdirectory must contain 'images' and 'annotations' subdirectories."
        )

    if not required_subdirs.issubset(set(val_subdirs)):
        raise ValueError(
            f"The 'val' subdirectory must contain 'images' and 'annotations' subdirectories."
        )


def load_config_object(params_file: str) -> ParamsConfig:
    """
    Load training parameters from a JSON file into a ParamsConfig object.
    Throws FileNotFoundError if the file does not exist.
    Throws Pydantic ValidationError if the JSON structure does not match ParamsConfig.
    """
    with open(params_file, "r") as f:
        params_dict = json.load(f)

    return ParamsConfig(**params_dict)


if __name__ == "__main__":
    app()
