import torch
from detectron2.data import detection_utils as utils, transforms as T


import copy

from coro_dt.augmentation.frame import FrameAugmentationWrapper


def validation_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # TODO: is this needed for validation? and shouldn't we use params from cfg?
    augmentations = [
        T.ResizeShortestEdge(
            short_edge_length=[800],
            max_size=1333,
            sample_style="choice",
        )
    ]

    aug_input = T.AugInput(image)
    transforms = T.AugmentationList(augmentations)(aug_input)
    image = aug_input.image

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


def build_custom_mapper(cfg):
    min_sizes = list(cfg.INPUT.MIN_SIZE_TRAIN)
    max_size = cfg.INPUT.MAX_SIZE_TRAIN

    def custom_mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        augmentations = [
            FrameAugmentationWrapper(
                frame_width_range=(0.05, 0.15),
                base_gray_range=(20, 50),
                noise_scale=15.0,
                method="random",
                crop_prob=0.5,
                p=0.5,
            ),
            T.ResizeShortestEdge(
                short_edge_length=min_sizes,
                max_size=max_size,
                sample_style="choice",
            ),
            T.RandomFlip(prob=0.0, horizontal=True, vertical=False),
            T.RandomRotation(angle=[-45, 45]),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
        ]
        aug_input = T.AugInput(image)
        transforms = T.AugmentationList(augmentations)(aug_input)
        image = aug_input.image
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    return custom_mapper
