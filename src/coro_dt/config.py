from enum import Enum
from pydantic import BaseModel


class Backbone(str, Enum):
    R_50 = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    R_101 = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"


POINTREND_WEIGHTS: dict[Backbone, str] = {
    Backbone.R_50: (
        "https://dl.fbaipublicfiles.com/detectron2/PointRend/"
        "InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/"
        "164955410/model_final_edd263.pkl"
    ),
    Backbone.R_101: (
        "https://dl.fbaipublicfiles.com/detectron2/PointRend/"
        "InstanceSegmentation/pointrend_rcnn_R_101_FPN_3x_coco/"
        "28119983/model_final_3f4d2a.pkl"
    ),
}


class ParamsConfig(BaseModel):
    base_lr: float = 0.00025
    anchor_sizes: list[list[int]] = [[16], [32], [64], [128], [256]]
    anchor_ratios: list[list[float]] = [[0.5, 1.0, 2.0]]
    freeze_at: int = 0
    roi_batch_size: int = 256
    input_min_sizes: list[int] = [640, 672, 704, 736, 768, 800]
    input_max_size: int = 1333
    backbone: Backbone = Backbone.R_50
    use_pointrend: bool = False
    roi_positive_fraction: float = 0.25
    rpn_nms_thresh: float = 0.7
    rpn_post_nms_topk_train: int = 2000


def apply_pointrend_overrides(cfg, num_classes: int = 1):
    """Apply PointRend-specific config overrides to a Detectron2 CfgNode."""
    cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
    cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = True
    cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
    cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = num_classes
    cfg.INPUT.MASK_FORMAT = "bitmask"
