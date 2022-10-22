import logging
import math
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn, distributions
import torch.nn.functional as F

# Detectron Imports
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, RetinaNetHead, permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes

# Project Imports
from src.probabilistic_modeling.modeling_utils import covariance_output_to_cholesky, clamp_log_variance, get_probabilistic_loss_weight

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.sigmoid(out1) - F.sigmoid(out2)))

@META_ARCH_REGISTRY.register()
class MCDRetinaNet(RetinaNet):
    """
    Adapting "Maximum Classifier Discrepancy for Unsupervised Domain Adaptation" for retinanet.
    https://arxiv.org/pdf/1712.02560.pdf
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != 'none'
        self.cls_var_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        self.bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        self.bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        if self.bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10

        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0

        self.current_step = 0
        if self.compute_bbox_cov:
            self.annealing_step = cfg.SOLVER.STEPS[1]

        # Define custom probabilistic head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.head_in_features]
        self.head = MCDRetinaNetHead(
            cfg,
            self.use_dropout,
            self.dropout_rate,
            self.compute_cls_var,
            self.compute_bbox_cov,
            self.bbox_cov_dims,
            feature_shapes)

        # Send to device
        self.to(self.device)

    def forward(
            self,
            batched_inputs,
            return_anchorwise_output=False,
            num_mc_dropout_runs=-1,
            real_last_percent=0.):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_anchorwise_output (bool): returns raw output for probabilistic inference

            real_last_percent(float): how many percentage (start from the end) of the current batch are real data, [0, 0.5, 1]

            num_mc_dropout_runs (int): perform efficient monte-carlo dropout runs by running only the head and
            not full neural network.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # Preprocess image
        images = self.preprocess_image(batched_inputs)

        # Extract features and generate anchors
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        anchors = self.anchor_generator(features)

        # MC_Dropout inference forward
        if num_mc_dropout_runs > 1:
            anchors = anchors * num_mc_dropout_runs
            features = features * num_mc_dropout_runs
            output_dict = self.produce_raw_output(anchors, features)
            return output_dict

        # Regular inference forward
        if return_anchorwise_output:
            return self.produce_raw_output(anchors, features)

        # Training and validation forward
        pred_logits1, pred_logits2, pred_anchor_deltas = self.head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits1 = [
            permute_to_N_HWA_K(
                x, self.num_classes) for x in pred_logits1]
        pred_logits2 = [
            permute_to_N_HWA_K(
                x, self.num_classes) for x in pred_logits2]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(
                x, 4) for x in pred_anchor_deltas]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [
                x["instances"].to(
                    self.device) for x in batched_inputs]

            gt_classes, gt_boxes = self.label_anchors(
                anchors, gt_instances)

            self.anchors = torch.cat(
                [Boxes.cat(anchors).tensor for i in range(len(gt_instances))], 0)

            # Loss is computed based on what values are to be estimated by the neural
            # network
            losses = self.losses(
                anchors,
                gt_classes,
                gt_boxes,
                pred_logits1,
                pred_logits2,
                pred_anchor_deltas,
                real_last_percent=real_last_percent)

            self.current_step += 1

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits1, pred_anchor_deltas, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            return losses
        else:
            results = self.inference(
                anchors,
                pred_logits1,
                pred_anchor_deltas,
                images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
            self,
            anchors,
            gt_classes,
            gt_boxes,
            pred_class_logits1,
            pred_class_logits2,
            pred_anchor_deltas,
            real_last_percent=0):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas`, `pred_class_logits_var` and `pred_bbox_cov`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_classes)
        num_real_images = int(num_images * real_last_percent)
        # print(num_images)
        # print(real_last_percent)
        # print(num_real_images)
        # print("pred_class_logits1:")
        # for p in pred_class_logits1:
        #     print(p.shape)
        # print("pred_class_logits2:", len(pred_class_logits2[0]))
        # print("gt_classes:", len(gt_classes))
        # print("gt_boxes:", len(gt_boxes))
        # print("pred_anchor_deltas:", len(pred_anchor_deltas[0]))

        # Transform per-feature layer lists to a single tensor
        pred_class_logits1 = cat(pred_class_logits1, dim=1)
        pred_class_logits2 = cat(pred_class_logits2, dim=1)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)

        if num_real_images > 0 and num_real_images < num_images:
            sim_idx = int(num_images - num_real_images)
            gt_classes = gt_classes[:sim_idx]
            gt_boxes = gt_boxes[:sim_idx]
            pred_class_logits_real1 = pred_class_logits1[sim_idx:]
            pred_class_logits_real2 = pred_class_logits2[sim_idx:]
            pred_class_logits1 = pred_class_logits1[:sim_idx]
            pred_class_logits2 = pred_class_logits2[:sim_idx]
            pred_anchor_deltas = pred_anchor_deltas[:sim_idx]

        elif num_real_images == num_images:
            prob1 = F.sigmoid(pred_class_logits1)
            prob2 = F.sigmoid(pred_class_logits2)
            loss_dis = torch.sum(torch.abs( prob1 - prob2)) / self.loss_normalizer
            return {"loss_cls": 0., "loss_dis":loss_dis, "loss_box_reg": 0.}

        gt_labels = torch.stack(gt_classes)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)

        gt_anchor_deltas = [
            self.box2box_transform.get_deltas(
                anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + \
            (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regression loss

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.

        gt_classes_target = torch.nn.functional.one_hot(
            gt_labels[valid_mask],
            num_classes=self.num_classes +
            1)[
            :,
            :-
            1].to(
            pred_class_logits1[0].dtype)  # no loss for the last (background) class

        # Classification losses
        # Standard loss computation in case one wants to use this code
        # without any probabilistic inference.
        loss_cls1 = sigmoid_focal_loss_jit(
            pred_class_logits1[valid_mask],
            gt_classes_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)
        loss_cls2 = sigmoid_focal_loss_jit(
            pred_class_logits2[valid_mask],
            gt_classes_target,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / max(1, self.loss_normalizer)
        loss_cls = loss_cls1 + loss_cls2
        
        if num_real_images > 0:
            prob1 = F.sigmoid(pred_class_logits_real1)
            prob2 = F.sigmoid(pred_class_logits_real2)
            loss_dis = torch.sum(torch.abs( prob1 - prob2)) / self.loss_normalizer
        else:
            loss_dis = 0.

        # Compute Regression Loss
        pred_anchor_deltas = pred_anchor_deltas[pos_mask]
        gt_anchors_deltas = gt_anchor_deltas[pos_mask]
        
        # Standard regression loss in case no variance is needed to be
        # estimated.
        loss_box_reg = smooth_l1_loss(
            pred_anchor_deltas,
            gt_anchors_deltas,
            beta=self.smooth_l1_beta,
            reduction="sum",
        ) / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_dis":loss_dis, "loss_box_reg": loss_box_reg}

    def produce_raw_output(self, anchors, features):
        """
        Given anchors and features, produces raw pre-nms output to be used for custom fusion operations.
        """
        # Perform inference run
        pred_logits, _, pred_anchor_deltas = self.head(
            features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(
                x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(
                x, 4) for x in pred_anchor_deltas]
        # Create raw output dictionary
        raw_output = {'anchors': anchors}

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.
        raw_output.update({'box_cls': pred_logits,
                           'box_delta': pred_anchor_deltas})
        return raw_output


class MCDRetinaNetHead(RetinaNetHead):
    """
    The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg,
                 use_dropout,
                 dropout_rate,
                 compute_cls_var,
                 compute_bbox_cov,
                 bbox_cov_dims,
                 input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

        # Extract config information
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims

        # For consistency all configs are grabbed from original RetinaNet
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        cls_subnet = []
        cls_subnet2 = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(nn.ReLU())

            cls_subnet2.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet2.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(nn.ReLU())

            if self.use_dropout:
                cls_subnet.append(nn.Dropout(p=self.dropout_rate))
                cls_subnet2.append(nn.Dropout(p=self.dropout_rate))
                bbox_subnet.append(nn.Dropout(p=self.dropout_rate))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.cls_subnet2 = nn.Sequential(*cls_subnet2)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels,
            num_anchors *
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.cls_score2 = nn.Conv2d(
            in_channels,
            num_anchors *
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)

        for modules in [
                self.cls_subnet,
                self.cls_subnet2,
                self.bbox_subnet,
                self.cls_score,
                self.cls_score2,
                self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)
        torch.nn.init.constant_(self.cls_score2.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.

            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.
        """
        logits1 = []
        logits2 = []
        bbox_reg = []
        for feature in features:
            logits1.append(self.cls_score(self.cls_subnet(feature)))
            logits2.append(self.cls_score2(self.cls_subnet2(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        return_vector = [logits1, logits2, bbox_reg]
        return return_vector
