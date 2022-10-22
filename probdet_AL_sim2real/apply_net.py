"""
Probabilistic Detectron Inference Script
"""
import os
import torch
from shutil import copyfile
# Detectron imports
from detectron2.engine import launch, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
# from detectron2.data import build_detection_test_loader, MetadataCatalog
# from detectron2.checkpoint import DetectionCheckpointer

# Project imports
import src.core
from src.core.setup import setup_config, setup_arg_parser
from src.probabilistic_inference.inference_utils import build_predictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # Setup config
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True,
                       setup_logger_flag=True)
                       
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.DEVICE = device.type
    cfg.DATASETS.TEST = [args.test_dataset]

    # Set up number of cpu threads#
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)
    EVAL_WITH_COCO = False
    cat_mapping_dict = None

    # Create inference output directory and copy inference config file to keep
    # track of experimental settings
    inference_output_dir = os.path.join(
        cfg['OUTPUT_DIR'],
        "evalutaion_temp",
        args.test_dataset,
        os.path.split(args.inference_config)[-1][:-5])

    os.makedirs(inference_output_dir, exist_ok=True)
    copyfile(args.inference_config, os.path.join(
        inference_output_dir, os.path.split(args.inference_config)[-1]))

    # Build predictor
    cfg.freeze()
    predictor = build_predictor(cfg)
    # test_data_loader = build_detection_test_loader(cfg, dataset_name=args.test_dataset)
     
    evaluators = [COCOEvaluator(args.test_dataset, cfg, True, inference_output_dir)]
    DefaultTrainer.test(cfg, predictor, evaluators=evaluators)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()
    # Support single gpu inference only.
    args.num_gpus = 1
    print("Command Line Args:")
    print(args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
