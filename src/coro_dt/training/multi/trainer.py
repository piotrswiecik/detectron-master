import datetime
import logging
import math
import os
import json
import time

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    DatasetCatalog,
    build_detection_train_loader,
    build_detection_test_loader,
)
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm
import mlflow
from dotenv import load_dotenv
import numpy as np
from coro_dt.data.adapter import Adapter
from coro_dt.training.multi.hooks import EvalHook, MLFlowHook
from coro_dt.training.multi.mappers import validation_mapper, build_custom_mapper


setup_logger()
load_dotenv()


class ArcadeTrainer(DefaultTrainer):
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
        hooks.append(EvalHook(self.cfg))
        hooks.append(MLFlowHook(self.cfg))
        return hooks


class ArcadeOrchestrator:
    def __init__(self, arcade_syntax_root: str, model_output_dir: str):
        self.log = logging.getLogger(__name__ + ".ArcadeOrchestrator")
        self.model_output_dir = model_output_dir

        self.class_names = []
        self.num_train_images = 0
        splits = ["train", "val"]

        for split in splits:
            json_file = os.path.join(
                arcade_syntax_root, split, "annotations", f"{split}.json"
            )
            img_dir = os.path.join(arcade_syntax_root, split, "images")

            if not os.path.exists(json_file):
                continue

            with open(json_file) as f:
                raw_data = json.load(f)

            adapter = Adapter(raw_data, img_dir)

            if split == "train":
                self.class_names = adapter.class_names
                self.num_train_images = len(adapter.as_list())
                print(
                    f"Training data loaded: {self.num_train_images} images, classes: {self.class_names}"
                )

            DatasetCatalog.register(f"arcade_{split}", lambda a=adapter: a.as_list())
            MetadataCatalog.get(f"arcade_{split}").set(
                thing_classes=adapter.class_names,
                id_reverse_map={v: k for k, v in adapter.id_map.items()},
                adapter_instance=adapter,
            )

            if self.num_train_images == 0:
                raise ValueError("No training images found")

            self.cfg = get_cfg()
            self.cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )

            self.cfg.DATASETS.TRAIN = ("arcade_train",)
            self.cfg.DATASETS.TEST = (
                ("arcade_val",) if "arcade_val" in DatasetCatalog.list() else ()
            )

            self.cfg.DATALOADER.NUM_WORKERS = 4
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )

            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.class_names)

            self.cfg.OUTPUT_DIR = self.model_output_dir

    def train(
        self,
        epochs: int,
        batch: int = 2,
        base_lr: float = 0.00025,
        hyperparameters: dict | None = None,
    ):
        hyperparameters = hyperparameters or {}

        one_epoch_iters = self.num_train_images // batch
        max_iter = one_epoch_iters * epochs

        self.cfg.SOLVER.IMS_PER_BATCH = batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.WARMUP_ITERS = 1000  # ramp up learning rate
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.8))
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters * 5

        self.cfg.TEST.EVAL_PERIOD = one_epoch_iters * 5

        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

        # backbone freeze (0, 1, 2)
        if "freeze_at" in hyperparameters and hyperparameters["freeze_at"] in [0, 1, 2]:
            self.cfg.MODEL.BACKBONE.FREEZE_AT = hyperparameters["freeze_at"]

        # anchor sizes
        if "anchor_sizes" in hyperparameters:
            sizes = hyperparameters["anchor_sizes"]
            if len(sizes) != 5:
                self.log.warning(
                    f"Warning: FPN expects 5 anchor sizes, got {len(sizes)}. This might crash."
                )
            self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = sizes

        # anchor aspect ratios
        if "anchor_ratios" in hyperparameters:
            self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = hyperparameters[
                "anchor_ratios"
            ]

        # roi head batch size
        if "roi_batch_size" in hyperparameters:
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = hyperparameters[
                "roi_batch_size"
            ]

        experiment_name = os.getenv("MLFLOW_EXPERIMENT")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(experiment_name)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with mlflow.start_run(
            run_name=f"detectron_epochs_{epochs}_batch_{batch}_dt_{timestamp}"
        ):
            trainer = ArcadeTrainer(self.cfg)
            trainer.resume_or_load(resume=False)

            self.log.info(f"Starting training for {epochs} epochs")
            trainer.train()
