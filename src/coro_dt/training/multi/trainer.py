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


setup_logger()
load_dotenv()
