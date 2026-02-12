from enum import Enum
from pydantic import BaseModel


class Backbone(str, Enum):
    R_50 = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    R_101 = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"


class ParamsConfig(BaseModel):
    base_lr: float = 0.00025
    anchor_sizes: list[list[int]] = [[16], [32], [64], [128], [256]]
    anchor_ratios: list[list[float]] = [[0.5, 1.0, 2.0]]
    freeze_at: int = 0
    roi_batch_size: int = 256
    input_min_sizes: list[int] = [640, 672, 704, 736, 768, 800]
    input_max_size: int = 1333
    backbone: Backbone = Backbone.R_50
