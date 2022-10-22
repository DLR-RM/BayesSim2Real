import os
import torch
import json
import tqdm
import numpy as np
from shutil import copyfile

from src.active_learning_src.active_selection import save_img_in_list
from src.probabilistic_inference.inference_utils import instances_to_json, build_predictor
from src.active_learning_src.utils.dataset_utils import (
    register_new_edan_coco_data, 
    register_new_ycbv_coco_data,
    )
from src.active_learning_src.active_learner import Active_learner

# detectron2
from detectron2.data import build_detection_test_loader

class Active_learner_light(Active_learner):
    def __init__(self, cfg, args):
        super().__init__(cfg, args)

    def _evalutate(self):
        """function to evaluate the trained model on a dataset, 
        in order to get uncertainty scores, different to the function 
        in parent class in the way of passing the predictions to selection
        stage in active learning, it's done on the fly without saving to 
        the disk here.
        """
        self.logger.info("############ Start Evaluation ############")
        sub_root_folder = self.cfg.OUTPUT_DIR 
        inference_output_dir = os.path.join(sub_root_folder,
                                            'inference' if self.cur_iter>1 else 'pretrained_inference',
                                            self.pool_set_name,
                                            os.path.split(self.inference_config)[-1][:-5])
        self.logger.info(f"############ Creating folder for Evaluation: {inference_output_dir} ############")
        os.makedirs(inference_output_dir, exist_ok=True)
        self.logger.info("############ We process the predictions in pool set on the fly! ############")
        if "RandomTopN" in self.cfg.AL.SAMPLING_MODE:
            # read current pool set
            # pool_json_file = MetadataCatalog.get(self.pool_set_name).json_file
            ori_pool_data_instances = json.load(open(self.pool_set_json_path, 'r'))
            image_id_list = [image['id'] for image in ori_pool_data_instances['images']]
            # randomly select a subset
            ratio_randomTop = self.cfg.AL.SUBSAMPLING_PERC
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
            
            if "edan" in self.cfg.DATASETS.TEST[0]:
                register_new_edan_coco_data(sub_pool_set_name, sub_pool_json_file, self.pool_set_image_folder)
            elif "ycbv" in self.cfg.DATASETS.TEST[0]:
                register_new_ycbv_coco_data(sub_pool_set_name, sub_pool_json_file, self.pool_set_image_folder)
            # update saved prediction file for selection in next iteration
            self.pred_json_path = os.path.join(inference_output_dir, "rnd_samples_coco_instances_results.json")
            test_data_loader = build_detection_test_loader(self.cfg, dataset_name=sub_pool_set_name)
             # start evaluation
            return self._eval_data_d2_predictor(test_data_loader, inference_output_dir)

        else:
            # update saved prediction file for selection in next iteration
            self.pred_json_path = os.path.join(inference_output_dir, "coco_instances_results.json")           
            test_data_loader = build_detection_test_loader(self.cfg, dataset_name=self.pool_set_name)
            # start evaluation
            return self._eval_data_d2_predictor(test_data_loader, inference_output_dir)
             
    def start_AL(self):
        """
        override this function to process predictions on the fly
        """
        
        test_results_cur_iter = []
        for iter in range(1, self.cfg.AL.NUM_ITER+1): 
            self.cur_iter = iter
            ###### part that has been changed ######
            pred_instances_list = self._evalutate()
            self._select_samples(pred_instances_list=pred_instances_list)
            del pred_instances_list
            ###### part that has been changed ######

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
        if "edan" in self.cfg.DATASETS.TEST[0]:
            self.pool_set_name = f'edan_pool_set_iter{iter_start}'
            register_new_edan_coco_data(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)
        elif "ycbv" in self.cfg.DATASETS.TEST[0]:
            self.pool_set_name = f'ycbv_pool_set_iter{iter_start}'
            register_new_ycbv_coco_data(self.pool_set_name, self.pool_set_json_path, self.pool_set_image_folder)

        for iter in range(iter_start, self.cfg.AL.NUM_ITER+1): 
            self.cur_iter = iter
            if iter > iter_start:
                ###### part that has been changed ######
                pred_instances_list = self._evalutate()
                self._select_samples(pred_instances_list=pred_instances_list)
                del pred_instances_list
                ###### part that has been changed ######
            # prepare new folders for logging re-training
            self.cfg.defrost()
            self.cfg.OUTPUT_DIR = os.path.join(self.al_exp_folder_path, f"iter{iter}")
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            self.cfg.freeze()
            mAP_dict = self._retrain()
            test_results_cur_iter.append({f"iter{iter}": mAP_dict})
            with open(os.path.join(self.al_exp_folder_path, self.results_json_filename), 'w') as f:
                json.dump(test_results_cur_iter, f)