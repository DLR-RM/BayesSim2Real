import numpy as np
import os
import random
import torch

from shutil import ExecError, copyfile

# Project imports
import src.core as core
from src.core.datasets.setup_datasets import setup_all_datasets
from src.probabilistic_modeling.probabilistic_retinanet import ProbabilisticRetinaNet
from src.probabilistic_modeling.probabilistic_generalized_rcnn import (
    ProbabilisticGeneralizedRCNN, 
    DropoutFastRCNNConvFCHead, 
    ProbabilisticROIHeads)
    
# Detectron imports
import detectron2.utils.comm as comm
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.engine import default_argument_parser, default_setup
from detectron2.utils.logger import setup_logger


def setup_arg_parser():
    """
    Sets up argument parser for python scripts.

    Returns:
        arg_parser (ArgumentParser): Argument parser updated with probabilistic detectron args.

    """
    arg_parser = default_argument_parser()

    arg_parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="path to dataset directory")

    arg_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed to be used for all scientific computing libraries")

    arg_parser.add_argument(
        "--al-config",
        type=str,
        default="",
        help="active learning parameter: Path to the inference config, which is different from training config. Check readme for more information.")
    arg_parser.add_argument(
        "--inference-config",
        type=str,
        default="",
        help="Inference parameter: Path to the inference config, which is different from training config. Check readme for more information.")

    arg_parser.add_argument(
        "--test-dataset",
        type=str,
        default="",
        help="Inference parameter: Dataset used for testing. Can be one of the following: 'coco_2017_custom_val', 'openimages_val', 'openimages_ood_val' ")

    return arg_parser


def add_customized_config(cfg):
    """
        Add configuration elements specific to probabilistic detectron.

    Args:
        cfg (CfgNode): detectron2 configuration node.

    """
    _C = cfg

    # add acitve learning settings
    _C.AL = CN()
    _C.AL.INITIAL_CONFIG = ''
    _C.AL.MODEL_TO_LOAD = '' # which model to load at the beginning of AL, such as 'model_0039999.pth, or 'model_final.pth"
    _C.AL.NUM_ITER = 10
    _C.AL.NUM_ACQ_EACH_ITER = 20
    _C.AL.VAL_DATASET = 'edan_real_val'
    _C.AL.MAX_EPOCH_EACH_ITER = 500 # number epochs
    _C.AL.EVAL_EVERY_EPOCH = 100 # number epochs
    _C.AL.SAMPLING_MODE = 'TopN' # 'TopN', 'Coreset', 'RandomTopN', 'TopNBalancedSyn', 'TopNBalancedReal', 'RandomTopNBalancedSyn', 'RandomTopNBalancedReal'
    _C.AL.ACQ_MODE = "cls" # 'cls', 'reg', 'both', 'max', "cls_bald", "cls_batch_bald"
    _C.AL.ACQ_CLS_BATCH_BALD = CN()
    _C.AL.ACQ_CLS_BATCH_BALD.NUM_SAMPLES = 1000
    _C.AL.AGG_MODE = 'max' # 'sum', 'avg', 'max'
    _C.AL.ACQ_SUM = CN()
    _C.AL.ACQ_SUM.WEIGHT_CLS = 1.0
    _C.AL.ACQ_SUM.WEIGHT_REG = 0.01
    _C.AL.SUBSAMPLING_PERC = 0.5 # percentage to be subsampled before uncertainty sampling

    # add to original config
    _C.MODEL.BACKBONE.FREEZE_ALL = False
    _C.MODEL.ROI_BOX_HEAD.DROPOUT_RATE = 0.0
    _C.INPUT.COPYPOSE_AUG = False
    _C.SOLVER.OPTIMIZER = "SGD"
    _C.SOLVER.ALPHA = 0.001
    
    # Probabilistic Modeling Setup
    _C.MODEL.PROBABILISTIC_MODELING = CN()
    _C.MODEL.PROBABILISTIC_MODELING.MC_DROPOUT = CN()
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS = CN()
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS = CN()

    # Annealing step for losses that require some form of annealing
    _C.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP = 0

    # Monte-Carlo Dropout Settings
    _C.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE = 0.0

    # Loss configs
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES = 3

    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE = 'diagonal'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES = 1000

    # Probabilistic Inference Setup
    _C.PROBABILISTIC_INFERENCE = CN()
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.BAYES_OD = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES = CN()
    _C.PROBABILISTIC_INFERENCE.DIST_NAME_MUTUAL_INFO = "categorical" # "bernoulli", "categorical"
    _C.PROBABILISTIC_INFERENCE.SAVE_PROB_VEC_SAMPLES = False
    _C.PROBABILISTIC_INFERENCE.RETURN_LOGITS = False
    _C.PROBABILISTIC_INFERENCE.ONLY_EPISTEMIC_COV = False

    # General Inference Configs
    _C.PROBABILISTIC_INFERENCE.INFERENCE_MODE = 'standard_nms'
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE = False
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS = 1
    _C.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD = 0.7

    # Bayes OD Configs
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.DIRCH_PRIOR = 'uniform'

    # Ensembles Configs
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE = 'pre_nms'
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS = [
        0, 1000, 2000, 3000, 4000]
    # 'mixture_of_gaussian' or 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE = 'mixture_of_gaussians'


def setup_config(args, random_seed=None, is_testing=False, setup_logger_flag=True):
    # Get default detectron config file
    cfg = get_cfg()
    # add_detr_config(cfg)
    add_customized_config(cfg)

    # Update default config file with custom config file
    # configs_dir = core.configs_dir()
    # args.config_file = os.path.join(configs_dir, args.config_file)
    cfg.merge_from_file(args.config_file)

    # Update config with inference configurations. Only applicable for when in
    # probabilistic inference mode.
    
    if args.inference_config != "":
        cfg.merge_from_file(args.inference_config)

    # Create output directory
    if cfg['OUTPUT_DIR'] is not None:
       exp_root = cfg['OUTPUT_DIR']
       cfg['OUTPUT_DIR'] = os.path.join(exp_root,
                                       os.path.split(os.path.split(args.config_file)[0])[-1],
                                       os.path.split(args.config_file)[-1][:-5])
    else:
        raise ExecError("cfg['OUTPUT_DIR'] is None, please specify one for experiment root folder!")

    if is_testing:
        if not os.path.isdir(cfg['OUTPUT_DIR']):
            raise NotADirectoryError(
                "Checkpoint directory {} does not exist.".format(
                    cfg['OUTPUT_DIR']))

    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)
    # copy config file to output directory
    copyfile(args.config_file, os.path.join(
        cfg['OUTPUT_DIR'], os.path.split(args.config_file)[-1]))

    # Freeze config file
    cfg['SEED'] = random_seed
    cfg.freeze()

    # Setup logger for probabilistic detectron module
    if setup_logger_flag:
        # Initiate default setup
        default_setup(cfg, args)
        setup_logger(
            output=cfg.OUTPUT_DIR,
            distributed_rank=comm.get_rank(),
            name="Detectron")

    # Set a fixed random seed for all numerical libraries
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Handle cases when this function has been called multiple times. In that case skip fully.
    # Todo this is very bad practice, should fix.
    try:
        setup_all_datasets()
        return cfg
    except AssertionError:
        return cfg


def al_setup_config(args, random_seed=None):
    # Get default detectron config file
    cfg = get_cfg()
    # add_detr_config(cfg)
    add_customized_config(cfg)

    # Update default config file with custom config file
    args.al_config = os.path.join(args.config_file_root, args.al_config)
    cfg.merge_from_file(args.al_config) # this has to be done first, otherwise no AL.INITIAL_CONFIG specified
    args.inference_config = os.path.join(args.config_file_root, args.inference_config)
    cfg.merge_from_file(args.inference_config)
    args.config_file = os.path.join(args.config_file_root, cfg.AL.INITIAL_CONFIG)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(args.al_config) # to make sure the training settings are in the latest config file!

    # Create output directory
    if cfg['OUTPUT_DIR'] is not None:
        exp_root = cfg['OUTPUT_DIR']
        cfg['OUTPUT_DIR'] = os.path.join(exp_root,
                                        os.path.split(os.path.split(args.config_file)[0])[-1],
                                        os.path.split(args.config_file)[-1][:-5])
    else:
        raise ExecError("cfg['OUTPUT_DIR'] is None, please specify one for experiment root folder!")

    os.makedirs(cfg['OUTPUT_DIR'], exist_ok=True)
    # copy config file to output directory
    copyfile(args.config_file, os.path.join(
        cfg['OUTPUT_DIR'], os.path.split(args.config_file)[-1]))

    # Freeze config file
    cfg['SEED'] = random_seed
    cfg.freeze()

    # Set a fixed random seed for all numerical libraries
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Handle cases when this function has been called multiple times. In that case skip fully.
    # Todo this is very bad practice, should fix.
    try:
        setup_all_datasets()
        return cfg
    except AssertionError:
        return cfg
