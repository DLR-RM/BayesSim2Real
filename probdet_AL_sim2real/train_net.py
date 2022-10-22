"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import src.core
import os
import sys
import matplotlib.pyplot as plt
# Detectron imports
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results

# Project imports
from src.core.setup import setup_config, setup_arg_parser
from src.core.datasets.dataset_mappers import DataAugDatasetMapper, RCNNDataAugDatasetMapper


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DatasetEvaluators object
        """
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Builds DataLoader for test set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode
            dataset_name(str): registered dataset name

        Returns:
            detectron2 DataLoader object specific to the test set.
        """
        return build_detection_test_loader(
            cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Builds DataLoader for train set.
        Args:
            cfg(CfgNode): a detectron2 CfgNode

        Returns:
            detectron2 DataLoader object specific to the train set.
        """
        if cfg.INPUT.COPYPOSE_AUG and 'RCNN' in cfg.MODEL.META_ARCHITECTURE:
            return build_detection_train_loader(cfg, mapper=RCNNDataAugDatasetMapper(cfg, True)) 
        elif cfg.INPUT.COPYPOSE_AUG and 'RetinaNet' in cfg.MODEL.META_ARCHITECTURE:
            policy_version = "policy_v001" # "policy_v001", "policy_v0", "policy_v3"
            return build_detection_train_loader(cfg, mapper=DataAugDatasetMapper(cfg, True, policy_version)) 
        else:
            return build_detection_train_loader(cfg) 


def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed)

    trainer = Trainer(cfg)
    # for checking augmentations
    # for idx, input_im in enumerate(trainer._trainer.data_loader):
    #     if idx < 5:
    #         for im_id in range(len(input_im)):
    #             img = input_im[im_id]['image'].permute(1, 2, 0)
    #             plt.imshow(img[..., [2,1,0]]) # BGR to RGB
    #             plt.show()
    #     else:
    #         exit

    if args.eval_only:
        model = trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
