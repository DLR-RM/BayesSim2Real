# from src.active_learning_src.active_learner import Active_learner
# from src.active_learning_src.active_learner_batch_bald import Active_learner_batch_bald
from src.probabilistic_inference.inference_utils import build_predictor
from src.active_learning_src.al_trainer import AL_Trainer
from src.core.setup import setup_config, setup_arg_parser
from detectron2.engine import launch

import json
import os

def main(args):
    # initial settings
    cfg = setup_config(args, random_seed=args.random_seed, setup_logger_flag=True)
    al_exp_folder_path = os.path.join(cfg.OUTPUT_DIR, args.al_exp_name)
    test_set = args.test_dataset # "ycbv_real_test_all"  
    num_iter = args.num_iter

    test_results_cur_iter = []
    for iter in range(1, num_iter+1): 
        cur_iter = iter
        cfg.defrost()
        # change output dir for loading corresponding weights
        cfg.OUTPUT_DIR = os.path.join(al_exp_folder_path, f"iter{iter}")
        assert os.path.exists(cfg.OUTPUT_DIR), f"{cfg.OUTPUT_DIR} does not exist!"
        cfg.freeze()
        print(f"############ Evaluating {test_set} for model in iter {cur_iter} ############")
        print(f"############ {cfg.OUTPUT_DIR} ############")
        predictor = build_predictor(cfg) # load the last checkpoint in `cfg.OUTPUT_DIR`(defined by a `last_checkpoint` file),
        results = AL_Trainer.test(cfg, predictor.model, test_set=test_set)
        print(f"############ Finish Evaluatiation in iter {cur_iter} ############")
        del predictor
        mAP_dict = results['bbox']
        test_results_cur_iter.append({f"iter{iter}": mAP_dict})
        with open(os.path.join(al_exp_folder_path, f"al_results_{test_set}.json"), 'w') as f:
            json.dump(test_results_cur_iter, f)

if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()
    arg_parser.add_argument(
        "--num_iter",
        type=int,
        default=10,
        help="number of iterations for active learning.")
    arg_parser.add_argument(
        "--al_exp_name",
        type=str,
        help="experiment name of al seetings.")
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
