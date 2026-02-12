import json
import logging
import math

import mlflow
import numpy as np
import torch
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

from coro_dt.data.converter import DetectronToArcadeConverter
from coro_dt.data.metrics import ArcadeMetricsCalculator
from coro_dt.training.multi.mappers import validation_mapper


class BinaryEvalHook(HookBase):
    """Evaluation hook for binary vessel detection."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.period = cfg.TEST.EVAL_PERIOD
        self.log = logging.getLogger(__name__)

        self.reverse_id_map = {0: 0}
        self.converter = DetectronToArcadeConverter(self.reverse_id_map)
        self.metrics_calculator = ArcadeMetricsCalculator()

        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self):
        """Load ground truth annotations from BinaryAdapter."""
        try:
            metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        except (IndexError, KeyError):
            self.log.warning("No test dataset configured. Metrics unavailable.")
            return {}

        if not hasattr(metadata, "adapter_instance"):
            self.log.warning("Adapter instance not found in metadata.")
            return {}

        adapter = metadata.adapter_instance

        gt_by_image = {}
        for item in adapter:
            image_id = item["image_id"]
            gt_anns = []
            for ann in item["annotations"]:
                for seg in ann["segmentation"]:
                    gt_anns.append({
                        "segmentation": seg,
                        "category_id": 0,
                    })
            gt_by_image[image_id] = {
                "annotations": gt_anns,
                "height": item["height"],
                "width": item["width"],
            }

        return gt_by_image

    def _build_data_loader(self):
        """Build data loader for evaluation."""
        return build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=validation_mapper,
        )

    def _run_evaluation(self):
        """Run evaluation and compute metrics."""
        data_loader = self._build_data_loader()

        model = self.trainer.model
        was_training = model.training

        losses = []
        all_ious = []
        all_dices = []
        total_matches = 0
        total_predictions = 0
        total_ground_truth = 0

        self.converter.reset_counter()

        for inputs in data_loader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.no_grad():
                model.train()
                loss_dict = model(inputs)
                losses.append(sum(loss for loss in loss_dict.values()))

                if self.ground_truth:
                    model.eval()
                    outputs = model(inputs)

                    for input_dict, output in zip(inputs, outputs):
                        image_id = input_dict["image_id"]

                        if image_id not in self.ground_truth:
                            continue

                        gt_data = self.ground_truth[image_id]
                        original_height = gt_data["height"]
                        original_width = gt_data["width"]

                        transformed_height = input_dict["image"].shape[1]
                        transformed_width = input_dict["image"].shape[2]

                        instances = output["instances"].to("cpu")

                        if len(instances) > 0 and instances.has("pred_masks"):
                            mask_height = instances.pred_masks.shape[1]
                            mask_width = instances.pred_masks.shape[2]
                        else:
                            mask_height = transformed_height
                            mask_width = transformed_width

                        pred_arcade = self.converter.convert_instances(
                            instances,
                            image_id,
                            original_height=original_height,
                            original_width=original_width,
                            transformed_height=mask_height,
                            transformed_width=mask_width,
                            score_threshold=0.5,
                        )

                        gt_arcade = gt_data["annotations"]

                        metrics = self.metrics_calculator.calculate_metrics_for_image(
                            pred_arcade,
                            gt_arcade,
                            original_height,
                            original_width,
                            iou_threshold=0.5,
                        )

                        all_ious.extend(metrics["ious"])
                        all_dices.extend(metrics["dices"])
                        total_matches += metrics["num_matches"]
                        total_predictions += metrics["num_predictions"]
                        total_ground_truth += metrics["num_ground_truth"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        if was_training:
            model.train()
        else:
            model.eval()

        mean_loss = torch.tensor(losses).mean().item()
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        print(f"Validation Loss: {mean_loss:.4f}")

        if self.ground_truth:
            mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
            mean_dice = float(np.mean(all_dices)) if all_dices else 0.0
            precision = total_matches / total_predictions if total_predictions > 0 else 0.0
            recall = total_matches / total_ground_truth if total_ground_truth > 0 else 0.0

            self.trainer.storage.put_scalar("binary/mean_iou", mean_iou)
            self.trainer.storage.put_scalar("binary/mean_dice", mean_dice)
            self.trainer.storage.put_scalar("binary/precision", precision)
            self.trainer.storage.put_scalar("binary/recall", recall)

            print(
                f"Binary Metrics - IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}, "
                f"Precision: {precision:.4f}, Recall: {recall:.4f} "
                f"(matches: {total_matches}, preds: {total_predictions}, gt: {total_ground_truth})"
            )

        comm.synchronize()

    def after_step(self):
        """Called after each training step."""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter

        if is_final or (self.period > 0 and next_iter % self.period == 0):
            self._run_evaluation()


class BinaryMLFlowHook(HookBase):
    """MLflow logging hook for binary training."""

    def __init__(self, cfg, log_period: int = 100):
        self.cfg = cfg
        self.log_period = log_period

    def before_train(self):
        params = {
            "task": "binary_vessel_detection",
            "num_classes": 1,
            "SOLVER.BASE_LR": self.cfg.SOLVER.BASE_LR,
            "SOLVER.MAX_ITER": self.cfg.SOLVER.MAX_ITER,
            "MODEL.ROI_HEADS.BATCH_SIZE": self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
            "DATASETS.TRAIN": str(self.cfg.DATASETS.TRAIN),
        }

        try:
            if hasattr(self.cfg.MODEL, "BACKBONE"):
                params["MODEL.BACKBONE.FREEZE_AT"] = self.cfg.MODEL.BACKBONE.FREEZE_AT
        except AttributeError:
            pass

        try:
            if hasattr(self.cfg.MODEL, "ANCHOR_GENERATOR"):
                params["MODEL.ANCHOR_GENERATOR.SIZES"] = json.dumps(
                    list(self.cfg.MODEL.ANCHOR_GENERATOR.SIZES)
                )
                params["MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"] = json.dumps(
                    list(self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS)
                )
        except AttributeError:
            pass

        mlflow.log_params(params)

    def after_step(self):
        if self.trainer.iter % self.log_period == 0:
            storage = self.trainer.storage
            metrics = {}

            for k in storage.latest().keys():
                if k in storage.histories():
                    val = storage.histories()[k].median(self.log_period)
                    if math.isfinite(val):
                        metrics[k] = val

            if metrics:
                mlflow.log_metrics(metrics, step=self.trainer.iter)

    def after_train(self):
        self.after_step()
