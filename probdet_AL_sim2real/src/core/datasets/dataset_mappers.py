import numpy as np
import PIL
import torch
import random
import copy
from PIL import Image, ImageEnhance, ImageFilter
from typing import Any, List, Optional, Tuple, Union
from torchvision.datasets import ImageFolder
from detectron2.structures import (
    Boxes,
    BoxMode)
from detectron2.data.dataset_mapper import DatasetMapper
import detectron2.data.detection_utils as utils
from detectron2.data.transforms import Transform
from detectron2.data.transforms.augmentation import AugInput, AugmentationList
import torchvision.transforms.functional as F
from .policy_v3_augmentations import Policy, ToTensor
from .pillow_augmentations import PillowTransform, to_pil, to_torch_uint8

class RCNNDataAugDatasetMapper(DatasetMapper):
    def __init__(self, 
                cfg, 
                is_train):
        super().__init__(cfg, 
                        is_train, 
                        augmentations=[PillowTransform()])

class Policy_AugInput(AugInput):
    def __init__(self,
                image,
                *,
                boxes: Optional[np.ndarray] = None,
                sem_seg: Optional[np.ndarray] = None,):
        # super().__init__(image, boxes=boxes, sem_seg=sem_seg)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg
    
    def transform(self, tfm: Transform) -> None:
        """
        In-place transform all attributes of this class.

        By "in-place", it means after calling this method, accessing an attribute such
        as ``self.image`` will return transformed data.
        """
        self.image, self.boxes = tfm.apply_image(self.image, self.boxes)
        # if self.boxes is not None:
        #     self.boxes = tfm.apply_box(self.boxes)
        if self.sem_seg is not None:
            self.sem_seg = tfm.apply_segmentation(self.sem_seg)

class Policy_Transform(Transform):
    def __init__(self, policy_version, post_transform):
        super().__init__()
        self._set_attributes(locals())
        self.policies = Policy(policy=policy_version, pre_transform=[], post_transform=post_transform, mag=9)

    def apply_image(self, img, box: np.ndarray):
        """
        Apply augmentations on the image(s).

        Args:
            img (PIL Image)
            box (np array)
        Returns:
            ndarray: augmented image(s).
        """
        # img = to_torch_uint8(np.ascontiguousarray(img))
        # img = F.to_tensor(img)
        img, box = self.policies(img, box)
        return img, box

    def apply_coords(self, coords: np.ndarray):
        return coords
        
class DataAugDatasetMapper(DatasetMapper):
    def __init__(self, 
                cfg, 
                is_train, 
                policy_version):
        super().__init__(cfg, 
                        is_train, 
                        augmentations=[Policy_Transform(policy_version, post_transform=[ToTensor()])])
        self.input_aug_flag = cfg.INPUT.COPYPOSE_AUG

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        # image_shape = image.shape[:2]  # h, w
        # utils.check_image_size(dataset_dict, image)
        # load with PIL image without convert_PIL_to_numpy() done in utils.read_image
        image = Image.open(dataset_dict["file_name"]).convert("RGB")
        if "width" not in dataset_dict:
            dataset_dict["width"] = image.size[0]
        if "height" not in dataset_dict:
            dataset_dict["height"] = image.size[0]
        image_shape = (image.size[1], image.size[0]) # h, w
        sem_seg_gt = None

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        # assume "annotations" is in dataset_dict
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # transform_instance_annotations
        boxes_gt = []
        annos = []
        for annotation in dataset_dict.pop("annotations"):
            if annotation.get("iscrowd", 0) == 0:
                bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
                # clip transformed bbox to image size
                # bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
                annotation["bbox"] = np.minimum(bbox, list(image_shape + image_shape)[::-1])
                annotation["bbox_mode"] = BoxMode.XYXY_ABS
                boxes_gt.append(annotation["bbox"])
                annos.append(annotation)
         
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        aug_input = Policy_AugInput(image, boxes=np.ascontiguousarray(boxes_gt))
        transforms = self.augmentations(aug_input)
        image, boxes_gt = aug_input.image, aug_input.boxes
        # cover to BGR which is the self.image_format
        # if image.shape[0] == 3:
        dataset_dict["image"] = image[[2,1,0],...] # image.permute(2, 0, 1)  
        # else:
        #     dataset_dict["image"] = image
        
        # update bboxes in annos:
        for idx, bbox in enumerate(boxes_gt):
            annos[idx]["bbox"] = np.ascontiguousarray(bbox)

        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
