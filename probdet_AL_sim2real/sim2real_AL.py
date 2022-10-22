from src.active_learning_src.active_learner import Active_learner
from src.active_learning_src.active_learner_light import Active_learner_light
from src.core.setup import al_setup_config, setup_arg_parser

from detectron2.engine import launch, default_argument_parser

def main(args):
    # initial settings
    cfg = al_setup_config(args, random_seed=args.random_seed)

    # add suffix if combination function is weighted sum
    if cfg.AL.ACQ_MODE == "both":
        if args.al_folder_suffix == "None":
            args.al_folder_suffix = ""
        args.al_folder_suffix += "_WCls{}".format(cfg.AL.ACQ_SUM.WEIGHT_CLS)
        args.al_folder_suffix += "_WReg{}".format(cfg.AL.ACQ_SUM.WEIGHT_REG)

    # init active learner
    if (cfg.AL.ACQ_MODE == "cls_batch_bald" and not args.random) or args.not_save_pred == 1:
        active_learner = Active_learner_light(cfg, args)
    else:
        active_learner = Active_learner(cfg, args)

    # start AL
    if args.start_from == 0:
        active_learner.start_AL()
    else:
        if cfg.AL.SAMPLING_MODE == "Coreset":
            active_learner.start_AL_from_iter_coreset(args.start_from)
        else:
            active_learner.start_AL_from_iter(args.start_from)

if __name__ == "__main__":
    # Create arg parser
    arg_parser = default_argument_parser()
    arg_parser.add_argument("--config-file-root", default="", metavar="FILE", help="root path to config file folders")
    arg_parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="random seed to be used for all scientific computing libraries")
    arg_parser.add_argument(
        "--random",
        action='store_true',
        help="flag for random selection")
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
        "--al_folder_suffix",
        type=str,
        default="None",
        help="al_folder_suffix.")
    arg_parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="from which iteration to start active learning.")
    arg_parser.add_argument(
        "--not_save_pred",
        type=int,
        default=1,
        help="flag to use the predictions for active learning on the fly without saving them on the disk.")
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
