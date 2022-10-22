import os
import sys
import torch
import json
import tqdm
import copy
import logging
import numpy as np
from shutil import copyfile

from src.active_learning_src.active_selection import select_save_informative_images, save_img_in_list
from src.probabilistic_inference.inference_utils import instances_to_json, build_predictor
from src.active_learning_src.utils.dataset_utils import (
    register_new_edan_coco_data, 
    register_new_ycbv_coco_data,
    )
from src.active_learning_src.al_trainer import AL_Trainer

# detectron2
from detectron2.engine import default_setup
from detectron2.data import build_detection_test_loader
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger

logger = setup_logger(name="Active_learner")

class Active_learner:
    def __init__(self, cfg, args):
        # parsing arguments
        self.inference_config = args.inference_config
        self.cfg = copy.deepcopy(cfg)
        self.random_seed = args.random_seed
        self.RANDOM = args.random # True
        self.NMS = "standard_nms" in args.inference_config
        
        # storage folder settings
        # assume self.cfg.OUTPUT_DIR will consist of AL learning folder and the base model
        self.cur_iter = 0 # updated in self.start_AL()
        self.root_folder = self.cfg.OUTPUT_DIR
        self.base_model_path = self.cfg.OUTPUT_DIR
        self.results_json_filename = "al_results.json"
        al_folder_suffix = args.al_folder_suffix
        if al_folder_suffix != "None":
            al_folder_suffix += "_OnlyTrnHeads" if self.cfg.MODEL.BACKBONE.FREEZE_ALL else ""
        else:
            al_folder_suffix = "_OnlyTrnHeads" if self.cfg.MODEL.BACKBONE.FREEZE_ALL else "None"
        self.al_exp_folder_path = os.path.join(self.root_folder, self._get_al_exp_folder_name(suffix=al_folder_suffix))
        self.pred_json_path = None # updated in self._evaluate()
        self.selected_data_folder = None # updated in self._select_samples()
        self.selected_sim_data_json_path = None
        self._init_pool_set() # currently specified by hand!!! updated in self._select_samples()
        if "edan" in self.cfg.DATASETS.TRAIN[0]:
            self.register_dataset_func = register_new_edan_coco_data
        elif "ycbv" in self.cfg.DATASETS.TRAIN[0]:
            self.register_dataset_func = register_new_ycbv_coco_data
        else:
            raise NotImplementedError(f"register_dataset_func for {self.cfg.DATASETS.TRAIN[0]} has NOT been implemented!")

        # modify config for defining Trainer for each iteration
        self.cfg.defrost()
        self.cfg.OUTPUT_DIR = self.al_exp_folder_path
        # important when batch blad is used!
        if self.cfg.AL.ACQ_MODE == "cls_batch_bald":
            self.cfg.PROBABILISTIC_INFERENCE.SAVE_PROB_VEC_SAMPLES = True 
        if "clue" in self.cfg.AL.SAMPLING_MODE or "Coreset" in self.cfg.AL.SAMPLING_MODE:
            self.cfg.PROBABILISTIC_INFERENCE.RETURN_LOGITS = True

        os.makedirs(self.al_exp_folder_path, exist_ok=True)
        # copy config file to output directory
        copyfile(args.al_config, os.path.join(self.al_exp_folder_path, os.path.split(args.al_config)[-1]))
        self.cfg.MODEL.WEIGHTS = os.path.join(self.base_model_path, self.cfg.AL.MODEL_TO_LOAD)
        self.cfg.freeze()

        default_setup(self.cfg, args)
        self.logger = logger # logging.getLogger("fvcore")
        # self.logger = logging.getLogger(__name__)

    def _get_al_exp_folder_name(self, suffix="None"):
        eval_num = self.cfg.AL.EVAL_EVERY_EPOCH if self.cfg.AL.EVAL_EVERY_EPOCH != 0 else self.cfg.TEST.EVAL_PERIOD
        if self.RANDOM: 
            al_exp_folder_name = (f"acq{self.cfg.AL.NUM_ACQ_EACH_ITER}_iter{self.cfg.AL.NUM_ITER}"
                f"_rnd{self.random_seed}_eval{eval_num}_lr{str(self.cfg.SOLVER.BASE_LR)}_RANDOM")
        else:
            if self.cfg.AL.ACQ_MODE == "cls_batch_bald" and "RandomTopN" not in self.cfg.AL.SAMPLING_MODE:
                al_exp_folder_name = (f"acq{self.cfg.AL.NUM_ACQ_EACH_ITER}_iter{self.cfg.AL.NUM_ITER}"
                    f"_{self.cfg.AL.ACQ_MODE}_{self.cfg.AL.ACQ_CLS_BATCH_BALD.NUM_SAMPLES}"
                    f"_rnd{self.random_seed}_eval{eval_num}_lr{str(self.cfg.SOLVER.BASE_LR)}")
            else:
                if "clue" in self.cfg.AL.SAMPLING_MODE:
                    al_exp_folder_name = (f"acq{self.cfg.AL.NUM_ACQ_EACH_ITER}_iter{self.cfg.AL.NUM_ITER}"
                        f"_{self.cfg.AL.ACQ_MODE}_{self.cfg.AL.SAMPLING_MODE}"
                        f"_rnd{self.random_seed}_eval{eval_num}_lr{str(self.cfg.SOLVER.BASE_LR)}")
                        
                elif "Coreset" in self.cfg.AL.SAMPLING_MODE:
                    al_exp_folder_name = (f"acq{self.cfg.AL.NUM_ACQ_EACH_ITER}_iter{self.cfg.AL.NUM_ITER}"
                        f"_{self.cfg.AL.SAMPLING_MODE}"
                        f"_rnd{self.random_seed}_eval{eval_num}_lr{str(self.cfg.SOLVER.BASE_LR)}")
                else:
                    al_exp_folder_name = (f"acq{self.cfg.AL.NUM_ACQ_EACH_ITER}_iter{self.cfg.AL.NUM_ITER}"
                        f"_{self.cfg.AL.AGG_MODE}_{self.cfg.AL.ACQ_MODE}_{self.cfg.AL.SAMPLING_MODE}"
                        f"_rnd{self.random_seed}_eval{eval_num}_lr{str(self.cfg.SOLVER.BASE_LR)}")
            if self.NMS:
                al_exp_folder_name += "_NMS"

        if self.cfg.AL.EVAL_EVERY_EPOCH == 0:
            al_exp_folder_name = al_exp_folder_name.replace(f"eval{self.cfg.AL.EVAL_EVERY_EPOCH}", f"ep{self.cfg.AL.MAX_EPOCH_EACH_ITER}")

        if suffix != "None":
            al_exp_folder_name += suffix

        return al_exp_folder_name

    def _init_pool_set(self):
        self.pool_set_name = self.cfg.DATASETS.TRAIN[0] # "edan_real_train" # "edan_real_train"
        self.pool_set_json_path = MetadataCatalog.get(self.pool_set_name).json_file 
        self.ori_pool_set_json_path = MetadataCatalog.get(self.pool_set_name).json_file 
        self.pool_set_image_folder = MetadataCatalog.get(self.pool_set_name).image_root 
        self.ycbv_sim_image_folder = MetadataCatalog.get("ycbv_sim").image_root  

    def _evalutate(self):
        """function to evaluate the trained model on a dataset, in order to get uncertainty scores
        """
        if self.RANDOM:
            self.logger.info("############ No need to evaluate because of RANDOM Selection ############")
            self.pred_json_path = None
        else:
            self.logger.info("############ Start Evaluation ############")
            sub_root_folder = self.cfg.OUTPUT_DIR 
            inference_output_dir = os.path.join(sub_root_folder,
                                                'inference' if self.cur_iter>1 else 'pretrained_inference',
                                                self.pool_set_name,
                                                os.path.split(self.inference_config)[-1][:-5])
            self.logger.info(f"############ Creating folder for Evaluation: {inference_output_dir} ############")
            os.makedirs(inference_output_dir, exist_ok=True)
            if "RandomTopN" in self.cfg.AL.SAMPLING_MODE:
                # read current pool set
                # pool_json_file = MetadataCatalog.get(self.pool_set_name).json_file
                ori_pool_data_instances = json.load(open(self.pool_set_json_path, 'r'))
                image_id_list = [image['id'] for image in ori_pool_data_instances['images']]
                # randomly select a subset
                ratio_randomTop = self.cfg.AL.SUBSAMPLING_PERC # 0.5, 1 - (0.5 - 0.05 * self.cur_iter)
                pre_selected_image_id_list = np.random.choice(image_id_list, size=int(len(image_id_list)*ratio_randomTop), replace=False)
                self.logger.info(f"############ Firstly randomly sampling {len(pre_selected_image_id_list)} for evaluation. ############")
                sub_pool_json_file = os.path.join(inference_output_dir, "pool_set_rnd_samples.json")
                sub_pool_set_name = self.pool_set_name + f"_rnd_samples_iter{self.cur_iter}"
                save_img_in_list(
                    source_json_file_or_ins=ori_pool_data_instances, 
                    save_json_file=sub_pool_json_file, 
                    img_id_list=pre_selected_image_id_list, 
                    in_list=True, 
                    logger=self.logger)
                
                self.register_dataset_func(sub_pool_set_name, sub_pool_json_file, self.pool_set_image_folder)
                # update saved prediction file for selection in next iteration
                self.pred_json_path = os.path.join(inference_output_dir, "rnd_samples_coco_instances_results.json")
                test_data_loader = build_detection_test_loader(self.cfg, dataset_name=sub_pool_set_name)

                # start evaluation
                final_output_list = self._eval_data_d2_predictor(test_data_loader, inference_output_dir)
                with open(self.pred_json_path, 'w') as fp:
                    json.dump(final_output_list, fp, indent=4, separators=(',', ': '))
                self.logger.info("############ Finished Evaluation ############")
                self.logger.info(f"############ Saving to {self.pred_json_path} ############")
            else:
                # update saved prediction file for selection in next iteration
                self.pred_json_path = os.path.join(inference_output_dir, "coco_instances_results.json")

                # start evaluation
                if (not os.path.exists(self.pred_json_path)):
                    self.logger.info("############ {} dose NOT exist! ############".format(self.pred_json_path))
                    test_data_loader = build_detection_test_loader(self.cfg, dataset_name=self.pool_set_name)
                    final_output_list = self._eval_data_d2_predictor(test_data_loader, inference_output_dir)
                    with open(self.pred_json_path, 'w') as fp:
                        json.dump(final_output_list, fp, indent=4, separators=(',', ': '))
                    self.logger.info("############ Finished Evaluation ############")
                    self.logger.info(f"############ Saving to {self.pred_json_path} ############")
                else:
                    self.logger.info("############ Evaluation Results Exist:  ############")
                    self.logger.info(self.pred_json_path)

    def _eval_data_d2_predictor(self, data_loader, inference_output_dir):
        copyfile(self.inference_config, os.path.join(
            inference_output_dir, os.path.split(self.inference_config)[-1]))
        # *************** DETECTRON2 part ***************
        # predictor output should be an instance datatype from DETECTRON2 structures
        # and contain the following fileds:
        # {"image_id", "category_id", "pred_boxes", "scores", "pred_classes", "pred_boxes_covariance"}
        # and bbox format is "xyxy" which will be converted to "xywh" in instances_to_json()
        predictor = build_predictor(self.cfg) # load weights in the current OUTPUT_DIR
        if self.cfg.AL.ACQ_MODE == "cls_batch_bald":
            to_cpu_numpy = False
        else:
            to_cpu_numpy = True
        final_output_list = []
        with torch.no_grad():
            with tqdm.tqdm(total=len(data_loader)) as pbar:
                for _, input_im in enumerate(data_loader):
                    outputs = predictor(input_im)
                    final_output_list.extend(
                        instances_to_json(outputs, input_im[0]['image_id'], to_cpu_numpy=to_cpu_numpy)
                        )
                    pbar.update(1)

        # release predictor
        del predictor
        # *************** DETECTRON2 part ***************
        return final_output_list

    def _select_samples(self, pred_instances_list=None):
        """function to select a subset of samples from a pool set for re-training the model
        """
        # update folder to save the selected data points and the fine-tuned model
        self.logger.info("############ Start AL Acquisition ############")
        # cfg.OUTPUT_DIR = os.path.join(al_exp_folder_path, f"iter{iter}")
        # create new folder for saving the selected data and pool set data
        sub_root_folder = self.cfg.OUTPUT_DIR
        self.selected_data_folder = os.path.join(sub_root_folder,
                                            'inference' if self.cur_iter>1 else 'pretrained_selected',
                                            'selected_json_files')
        os.makedirs(self.selected_data_folder, exist_ok=True)

        self.selected_data_json_path, self.pool_set_json_path, self.selected_sim_data_json_path = \
            select_save_informative_images(self.cfg,
                                            self.pred_json_path if pred_instances_list is None else pred_instances_list, 
                                            self.pool_set_json_path, 
                                            cur_iter=self.cur_iter,
                                            save_folder=self.selected_data_folder, 
                                            NMS=self.NMS,
                                            random=self.RANDOM,
                                            prev_selected_data_path=self.selected_data_json_path if self.cur_iter>1 else None)
        
        # register new remianing pool set
        if "edan" in self.cfg.DATASETS.TRAIN[0]:
            self.pool_set_name = f'edan_pool_set_iter{self.cur_iter}'
            self.register_dataset_func(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)
        elif "ycbv" in self.cfg.DATASETS.TRAIN[0]:
            self.pool_set_name = f'ycbv_pool_set_iter{self.cur_iter}'
            self.register_dataset_func(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)
        self.logger.info("############ Finished AL Acquisition ############")

    def _set_up_finetuning(self, trn_set_name, trn_set_2_name=None):
        self.cfg.defrost()
        cur_trn_set_size = len(DatasetCatalog.get(trn_set_name))
        num_epochs = self.cfg.AL.MAX_EPOCH_EACH_ITER
        batch_size = self.cfg.SOLVER.IMS_PER_BATCH
        eval_periord = self.cfg.AL.EVAL_EVERY_EPOCH
        
        if num_epochs > 0:
            self.cfg.SOLVER.MAX_ITER = int((cur_trn_set_size/batch_size) * (num_epochs + self.cur_iter*10))
        else:
            num_epochs = int(self.cfg.SOLVER.MAX_ITER * batch_size/cur_trn_set_size)
        self.cfg.CHECKPOINT_PERIOD = int(self.cfg.SOLVER.MAX_ITER)
        self.cfg.SOLVER.STEPS = (int(self.cfg.SOLVER.MAX_ITER/2), int(self.cfg.SOLVER.MAX_ITER/1.5))
        if eval_periord > 0:
            self.cfg.TEST.EVAL_PERIOD = int((cur_trn_set_size/batch_size) * eval_periord )
        else:
            eval_periord = int(self.cfg.TEST.EVAL_PERIOD * batch_size/cur_trn_set_size)
        self.logger.info(f"####### Training set ({trn_set_name}) in current iter {self.cur_iter} has size {cur_trn_set_size} #######")
        self.logger.info(f"####### Number of max training iter: {self.cfg.SOLVER.MAX_ITER} ({num_epochs} epochs), "
            f"with early stopping check after every {self.cfg.TEST.EVAL_PERIOD} iter ({eval_periord} epochs)! #######")
        
        # always fine-tune the synthetically trained model with the new selected dataset
        if "BalancedSyn" in self.cfg.AL.SAMPLING_MODE and trn_set_2_name is not None:
            self.cfg.DATASETS.TRAIN = [trn_set_name, trn_set_2_name]
        else:
            self.cfg.DATASETS.TRAIN = [trn_set_name]
        self.cfg.freeze()

    def _retrain(self):
        """function to retrain the model based on selected data
        """
        # register new selected dataset for retraining baesd on selected_data_json_path, pool_set_image_folder
        if "edan" in self.cfg.DATASETS.TRAIN[0]:
            trn_set_name = f'edan_real_iter{self.cur_iter}'
            self.register_dataset_func(trn_set_name, self.selected_data_json_path, self.pool_set_image_folder)
            # the training epoch increases with iter number linearly
            self._set_up_finetuning(trn_set_name)
        elif "ycbv" in self.cfg.DATASETS.TRAIN[0]:
            trn_set_name = f'ycbv_real_iter{self.cur_iter}'
            self.register_dataset_func(trn_set_name, self.selected_data_json_path, self.pool_set_image_folder)
            if "BalancedSyn" in self.cfg.AL.SAMPLING_MODE:
                sim_trn_set_name = f'ycbv_sim_iter{self.cur_iter}'
                self.register_dataset_func(sim_trn_set_name, self.selected_sim_data_json_path, self.ycbv_sim_image_folder)
            else: 
                sim_trn_set_name = None
            self._set_up_finetuning(trn_set_name, trn_set_2_name=sim_trn_set_name)

        self.logger.info("############ Start Fine-tuning ############")
        self.logger.info("####### Saving models and predictions in {} #######".format(self.cfg.OUTPUT_DIR))
        # ********** DETECTRON2 part **********
        trainer = AL_Trainer(self.cfg, test_set=self.cfg.DATASETS.TEST[0], val_set=self.cfg.AL.VAL_DATASET)
        trainer.resume_or_load(resume=False) # load from self.cfg.MODEL.WEIGHTS
        trainer.train()
        results = AL_Trainer.test(self.cfg, trainer.model, test_set=self.cfg.DATASETS.TEST[0])
        # ********** DETECTRON2 part **********
        # saved mAP of each iter for easy accessibility
        # mAP_dict = {}
        # for k, v in trainer.storage.latest().items():
        #     if 'bbox' in k:
        #         mAP_dict.update({k:v})
        self.logger.info("############ Finished Fine-tuning ############")
        del trainer
        return results['bbox']

    def start_AL(self):
        """ function to start active learning loop: 
        1. evaluate and get scores from pool set;
        2. select a subset of data from pool set based on the chosen sampling strategy;
        3. retrain the model based on the selected dataset;
        """
        
        test_results_cur_iter = []
        for iter in range(1, self.cfg.AL.NUM_ITER+1): 
            self.logger.info(f"############ Starting in iteration {iter} ############")
            self.cur_iter = iter
            self._evalutate()
            self._select_samples()
            # prepare new folders for logging re-training
            self.cfg.defrost()
            self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter}")
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            self.cfg.freeze()
            mAP_dict = self._retrain()
            test_results_cur_iter.append({f"iter{iter}": mAP_dict})
            with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'w') as f:
                json.dump(test_results_cur_iter, f)
            self.logger.info(f"############ Finished in iteration {iter} ############")
    
    def start_AL_from_iter_coreset(self, iter_start):
        """ function to start active learning loop from a specified iteration for coreset
        """
        assert iter_start > 1 
        self.cur_iter = iter_start
        # self.al_exp_folder_path = os.path.join(self.root_folder, start_al_folder)
        self.cfg.defrost()
        # set up the model path current iter for evalutation 
        self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter_start-1}")
        self.cfg.freeze()
        # set up paths of remaining poolset and selected data points of current iter for selecting samples
        if iter_start-1 > 1:
            prev_selected_data_folder = os.path.join(os.path.join(self.al_exp_folder_path, f"iter{iter_start-1}"),
                                                'inference',
                                                'selected_json_files')
        else:
            prev_selected_data_folder = os.path.join(os.path.join(self.al_exp_folder_path, "pretrained_selected"),
                                                'selected_json_files')
        self.selected_data_json_path = os.path.join(prev_selected_data_folder, f'selected_data_iter{iter_start-1}.json')
        self._select_samples()

        with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'r') as f:
            test_results_cur_iter = json.load(f)

        for iter in range(iter_start, self.cfg.AL.NUM_ITER+1): 
            self.cur_iter = iter
            if iter > iter_start:
                self._evalutate()
                self._select_samples()
            # prepare new folders for logging re-training
            self.cfg.defrost()
            self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter}")
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            self.cfg.freeze()
            mAP_dict = self._retrain()
            test_results_cur_iter.append({f"iter{iter}": mAP_dict})
            with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'w') as f:
                json.dump(test_results_cur_iter, f)

    def start_AL_from_iter(self, iter_start):
        """ function to start active learning loop from a specified iteration: 
        1. evaluate and get scores from pool set with model loaded from model_path of one iter before iter_start;
        2. select a subset of data from pool set based on chosen strategy;
        3. retrain the model based on the selected dataset;
        """
        
        assert iter_start > 1
        # self.al_exp_folder_path = os.path.join(self.root_folder, start_al_folder)
        self.cfg.defrost()
        # set up the model path current iter for evalutation 
        self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter_start}")
        self.cfg.freeze()

        with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'r') as f:
            test_results_cur_iter = json.load(f)

        # set up paths of remaining poolset and selected data points of current iter for selecting samples
        prev_selected_data_folder = os.path.join(os.path.join(self.al_exp_folder_path, f"iter{iter_start-1}"),
                                            'inference',
                                            'selected_json_files')
        self.selected_data_json_path = os.path.join(prev_selected_data_folder, f'selected_data_iter{iter_start}.json')
        self.pool_set_json_path = os.path.join(prev_selected_data_folder, f'pool_set_iter{iter_start}.json')

        # update pool set name of current iter for evalutation 
        if "edan" in self.cfg.DATASETS.TRAIN[0]:
            self.pool_set_name = f'edan_pool_set_iter{iter_start}'
            self.register_dataset_func(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)
        elif "ycbv" in self.cfg.DATASETS.TRAIN[0]:
            self.pool_set_name = f'ycbv_pool_set_iter{iter_start}'
            self.register_dataset_func(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)

        for iter in range(iter_start, self.cfg.AL.NUM_ITER+1): 
            self.cur_iter = iter
            if iter > iter_start:
                self._evalutate()
                self._select_samples()
            # prepare new folders for logging re-training
            self.cfg.defrost()
            self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter}")
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            self.cfg.freeze()
            mAP_dict = self._retrain()
            test_results_cur_iter.append({f"iter{iter}": mAP_dict})
            with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'w') as f:
                json.dump(test_results_cur_iter, f)