import datetime
import json
import logging
import os

import mlflow
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from dotenv import load_dotenv

from coro_dt.config import ParamsConfig
from coro_dt.data.binary_adapter import BinaryAdapter
from coro_dt.training.binary.hooks import BinaryEvalHook, BinaryMLFlowHook
from coro_dt.training.multi.mappers import build_custom_mapper


setup_logger()
load_dotenv()


class BinaryTrainer(DefaultTrainer):
    """Trainer for binary vessel detection."""

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=build_custom_mapper(cfg))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(BinaryEvalHook(self.cfg))
        hooks.append(BinaryMLFlowHook(self.cfg))
        return hooks


class BinaryOrchestrator:
    """Orchestrator for binary vessel detection training."""

    def __init__(
        self,
        arcade_syntax_root: str,
        model_output_dir: str,
        params: ParamsConfig,
        weights: str | None = None,
    ):
        self.log = logging.getLogger(__name__ + ".BinaryOrchestrator")
        self.model_output_dir = model_output_dir
        self.params = params
        self.backbone = params.backbone.value

        self.class_names = ["vessel"]
        self.num_train_images = 0
        splits = ["train", "val"]

        for split in splits:
            json_file = os.path.join(
                arcade_syntax_root, split, "annotations", f"{split}.json"
            )
            img_dir = os.path.join(arcade_syntax_root, split, "images")

            if not os.path.exists(json_file):
                self.log.warning(f"Annotation file not found: {json_file}")
                continue

            with open(json_file) as f:
                raw_data = json.load(f)

            adapter = BinaryAdapter(raw_data, img_dir)

            if split == "train":
                self.num_train_images = len(adapter.as_list())
                print(
                    f"Binary training data loaded: {self.num_train_images} images, "
                    f"classes: {self.class_names}"
                )

            dataset_name = f"arcade_binary_{split}"

            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)

            DatasetCatalog.register(dataset_name, lambda a=adapter: a.as_list())
            MetadataCatalog.get(dataset_name).set(
                thing_classes=self.class_names,
                id_reverse_map={0: 0},
                adapter_instance=adapter,
            )

        if self.num_train_images == 0:
            raise ValueError("No training images found")

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(self.backbone))

        self.cfg.DATASETS.TRAIN = ("arcade_binary_train",)
        self.cfg.DATASETS.TEST = (
            ("arcade_binary_val",)
            if "arcade_binary_val" in DatasetCatalog.list()
            else ()
        )

        self.cfg.DATALOADER.NUM_WORKERS = 4

        if weights is not None:
            self.log.info(f"Using custom weights: {weights}")
            self.cfg.MODEL.WEIGHTS = weights
        else:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.backbone)

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        self.cfg.OUTPUT_DIR = self.model_output_dir

    def train(
        self,
        epochs: int,
        batch: int = 2,
    ):
        """Run training."""
        p = self.params

        one_epoch_iters = self.num_train_images // batch
        max_iter = one_epoch_iters * epochs

        self.cfg.SOLVER.IMS_PER_BATCH = batch
        self.cfg.SOLVER.BASE_LR = p.base_lr
        self.cfg.SOLVER.WARMUP_ITERS = 1000
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters * 5

        self.cfg.TEST.EVAL_PERIOD = one_epoch_iters * 5

        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

        if p.freeze_at in (0, 1, 2):
            self.cfg.MODEL.BACKBONE.FREEZE_AT = p.freeze_at

        if len(p.anchor_sizes) != 5:
            self.log.warning(
                f"FPN expects 5 anchor sizes, got {len(p.anchor_sizes)}. This might crash."
            )
        self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = p.anchor_sizes
        self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = p.anchor_ratios
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = p.roi_batch_size
        self.cfg.INPUT.MIN_SIZE_TRAIN = tuple(p.input_min_sizes)
        self.cfg.INPUT.MAX_SIZE_TRAIN = p.input_max_size

        experiment_name = os.getenv("MLFLOW_EXPERIMENT", "detectron-binary")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"binary_epochs_{epochs}_batch_{batch}_dt_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            trainer = BinaryTrainer(self.cfg)
            trainer.resume_or_load(resume=False)

            self.log.info(f"Starting binary training for {epochs} epochs")
            trainer.train()
