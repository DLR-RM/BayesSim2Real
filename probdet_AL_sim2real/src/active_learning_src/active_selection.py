#!/usr/bin/python3

import os
import funcy 
import copy
import json
import logging
import tqdm
import numpy as np
import torch
from detectron2.data.catalog import MetadataCatalog
from detectron2.utils.logger import setup_logger


from src.active_learning_src.utils.utils import (
    compute_instance_surprise,   
    save_img_in_list,
    count_class_statistic
)
from src.active_learning_src.utils.sampling_methods import (
    BalancedReal_sampling,
    TopN_sampling,
    BalancedSyn_sampling,
    sample_img_coreset,
    sample_img_clue,
    sample_img_batch_bald
)

use_cuda = torch.cuda.is_available()
global_logger = setup_logger(name="Active_selection")

print(f"use_cuda: {use_cuda}")

device = "cuda" if use_cuda else "cpu"

def sample_img_based_on_img_unc(sampling_mode, 
                                num_select, 
                                img_unc_dict, 
                                sampling_hyper_param_dict, 
                                cur_pool_set_instances_gt,
                                logger):
    # ["TopN", "RandomTopN", "TopNBalancedSyn", "TopNBalancedReal", "RandomTopNBalancedSyn", "RandomTopNBalancedReal"
    if sampling_mode in ["TopNBalancedSyn", "RandomTopNBalancedSyn"]:
        num_class = sampling_hyper_param_dict[sampling_mode]["num_class"]
        selected_sim_data_json_path = sampling_hyper_param_dict[sampling_mode]["selected_sim_data_json_path"]
        sim_dataset_name = 'ycbv_sim' # "edan_sim", "ycbv_sim"
        selected_image_id_list = TopN_sampling(img_unc_dict, num_select)
        # select synthetic images to balance the dataset as much as possible
        # consider selected samples in previous iterations
        # if prev_selected_data_path is not None:
        #     selected_instances = json.load(open(prev_selected_data_path, 'r')) 
        #     logger.info(f"############ Accumulating {len(selected_instances['images'])} from previous iterations before selecting Sim Images from {prev_selected_data_path}. ############")
        #     for img in selected_instances['images']:
        #         temp_img_id_list.append(img['id'])
        
        cls_hist = count_class_statistic(num_class, 
                                        cur_pool_set_instances_gt, 
                                        selected_image_id_list, 
                                        logger, 
                                        descrp="Before RandomTopNBalancedSyn:")
                    
        num_selected_sim = int(len(selected_image_id_list)/2)
        selected_sim_img_id_list = BalancedSyn_sampling(num_selected_sim, 
                                                        cls_hist, 
                                                        logger=logger,
                                                        sim_dataset_name=sim_dataset_name)

        sim_json_file = MetadataCatalog.get(sim_dataset_name).json_file
        count_class_statistic(num_class, 
                            cur_pool_set_instances_gt, 
                            selected_sim_img_id_list, 
                            logger, 
                            descrp="After RandomTopNBalancedSyn:")
        # save selected synthetic images
        save_img_in_list(
            source_json_file_or_ins=sim_json_file, 
            save_json_file=selected_sim_data_json_path, 
            img_id_list=selected_sim_img_id_list, 
            in_list=True, # save images that are in the list of selected_sim_img_id_list
            logger=logger)
    elif sampling_mode in ["TopNBalancedReal", "RandomTopNBalancedReal"]:
        num_class = sampling_hyper_param_dict[sampling_mode]["num_class"]
        selected_image_id_list = TopN_sampling(img_unc_dict, num_select)
        # select pool images to balance the dataset as much as possible
        cls_hist = count_class_statistic(num_class, 
                                        cur_pool_set_instances_gt, 
                                        selected_image_id_list, 
                                        logger, 
                                        descrp=f"Before {sampling_mode}:")
        # selected_image_id_list get accumulated inside BalancedPseudoLabel_sampling
        selected_image_id_list = BalancedReal_sampling(selected_image_id_list, 
                                                            cls_hist, 
                                                            cur_pool_set_instances_gt, 
                                                            logger)
    elif sampling_mode == "RandomTopNBalancedSofmaxFeat":
        raise NotImplementedError
    elif sampling_mode == "TopN" or sampling_mode == "RandomTopN":
        selected_image_id_list = TopN_sampling(img_unc_dict, num_select)
    elif sampling_mode == "Hybrid":
        rdn_ratio = sampling_hyper_param_dict[sampling_mode]['random_ratio']
        image_id_list = [image['id'] for image in cur_pool_set_instances_gt['images']]
        selected_image_id_list = np.random.choice(image_id_list, size=int(num_select*rdn_ratio), replace=False).tolist()
        # selected_image_id_list = np.random.choice(list(img_unc_dict.keys()), size=int(num_select*rdn_ratio), replace=False).tolist()
        img_unc_dict_temp = copy.deepcopy(img_unc_dict)
        for im_id in selected_image_id_list:
            img_unc_dict_temp.pop(im_id)
        image_id_rank_list = sorted(img_unc_dict_temp, key=img_unc_dict_temp.get, reverse=True)
        num_unc_selection = int(np.ceil(num_select*(1-rdn_ratio)))
        selected_image_id_list.extend(image_id_rank_list[:num_unc_selection])
    else:
        raise NotImplementedError(f"{sampling_mode} has NOT been implemented!")

    return selected_image_id_list


def compute_inst_surprise_images(predicted_instances, acquisition_mode, NMS=False, weights=(1,1)):
    img_unc_dict = {} # key is image_id, value is a list of detections uncertainty
    # compute informativeness for each detection on each image
    with tqdm.tqdm(total=len(predicted_instances)) as pbar:
        for instance in predicted_instances:
            if NMS:
                instance["acqusition_dict"] = {
                    "cls": compute_instance_surprise(instance, mode="cls")
                }
                surprise = instance["acqusition_dict"]["cls"]
            else:
                instance["acqusition_dict"] = {acquisition_mode: compute_instance_surprise(instance, 
                                                                                            mode=acquisition_mode,
                                                                                            weight_cls=weights[0],
                                                                                            weight_reg=weights[1])}
                surprise = instance["acqusition_dict"][acquisition_mode]

            # append them to the corresponding images
            if instance['image_id'] in img_unc_dict.keys():
                img_unc_dict[instance['image_id']].append(surprise)
            else:
                img_unc_dict.update({instance['image_id']: [surprise]})
            pbar.update(1)
    # json.dump(predicted_instances, open(pred_json_path, 'w'), indent=4,separators=(',', ': '))
    return img_unc_dict


def aggregate_inst_each_image(aggregate_mode, img_unc_dict):
    # aggregate surprise of each detecton to the surprise of one image
    # aggregate predicted category information from detected instances
    if aggregate_mode == 'sum':
        for image_id in img_unc_dict.keys():
            img_unc_dict[image_id] = sum(img_unc_dict[image_id])
    elif aggregate_mode == 'avg':
        for image_id in img_unc_dict.keys():
            img_unc_dict[image_id] = np.average(img_unc_dict[image_id])
    elif aggregate_mode == 'max':
        for image_id in img_unc_dict.keys():
            img_unc_dict[image_id] = np.max(img_unc_dict[image_id])
    
    return img_unc_dict

def select_save_informative_images( cfg,
                                    pred_json_path_or_inst, 
                                    cur_pool_set_json, 
                                    save_folder, 
                                    cur_iter,
                                    random=False,
                                    NMS=False,
                                    prev_selected_data_path=None,):
    ''' function to select informative images from the pool set based on predictions saved in pred_json_path,
        and save two json files under the save_folder, the first one is the json file of the selected data,
        with name f'selected_data_{save_suffix}.json', the second is the file of the rest of data in the pool
        set with name f"pool_set_{save_suffix}.json"

        Args:
            num_select: number of images to select from the pool set;
            pred_json_path_or_inst:json file path of saved pool set predictions or list of instances in detetrcon2 format; 
            pool_set_json: ground truth json file of pool set data of current iteration; 
            save_folder: folder to save json files of selected data and remaining pool set
            aggregate_mode: way to produce the informativeness of an image by combining those of detectons on this iamge, ['sum', 'avg', 'max];
            acquisition_mode: acquisition function used to select data, {"both", "reg", "cls", "cls_bald", "cls_batch_bald", 'max'};
            sampling_mode: mode to sample num_select images based on calculated scores, {'TopN', 'Coreset', "RandomTopN", "RandomTopNBalancedSyn"};
            random: flag of whether to perform random selection;
            prev_selected_data_path: must no be None if accumulate is True;
            save_suffix: suffix added to saved json files;

        Return:
            selected_data_path;
            new_pool_set_path;
            selected_sim_data_json_path;
    '''
    save_suffix=f'iter{cur_iter}'
    num_select = int(cfg.AL.NUM_ACQ_EACH_ITER)
    aggregate_mode = cfg.AL.AGG_MODE
    acquisition_mode = cfg.AL.ACQ_MODE
    sampling_mode = cfg.AL.SAMPLING_MODE
    num_samples_batch_bald = int(cfg.AL.ACQ_CLS_BATCH_BALD.NUM_SAMPLES) # 1000
    weight_cls = float(cfg.AL.ACQ_SUM.WEIGHT_CLS)
    weight_reg = float(cfg.AL.ACQ_SUM.WEIGHT_REG)

    logger = global_logger # logging.getLogger("fvcore")
    selected_sim_data_json_path = None 
    cur_pool_set_instances_gt = json.load(open(cur_pool_set_json, 'r'))
    if random:
        logger.info(f"############ RANDOM Selection over pool set ({len(cur_pool_set_instances_gt['images'])}) ############")
        image_id_list = [image['id'] for image in cur_pool_set_instances_gt['images']]
        selected_image_id_list = np.random.choice(image_id_list, size=num_select, replace=False).tolist()
    else:
        if isinstance(pred_json_path_or_inst, str):
            predicted_instances = json.load(open(pred_json_path_or_inst, 'r'))
        else:
            predicted_instances = pred_json_path_or_inst

        # start querying data for annotations                                    
        if acquisition_mode == "cls_batch_bald":
            logger.info(f"############ Scoring predictions (with query function: {acquisition_mode}) over pool set. ############")
            selected_image_id_list = sample_img_batch_bald(predicted_instances, 
                                                            num_select, 
                                                            num_samples=num_samples_batch_bald)
        elif "clue" in sampling_mode:
            logger.info(f"############ Scoring predictions (with sampling mode:{sampling_mode}, query function: {acquisition_mode}) over pool set. ############")
            # update predicted_instances with computed scores
            compute_inst_surprise_images(predicted_instances, 
                                        acquisition_mode, 
                                        NMS=NMS, 
                                        weights=(weight_cls, weight_reg))
            selected_image_id_list = sample_img_clue(predicted_instances, 
                                                    num_select, 
                                                    acquisition_mode)
        elif "Coreset" in sampling_mode:
            logger.info(f"############ Scoring predictions (with sampling mode:{sampling_mode}, query function: {acquisition_mode}) over pool set. ############")
            selected_image_id_list, img_feat_dict = sample_img_coreset(predicted_instances, 
                                                                        num_select, 
                                                                        prev_selected_data_path,
                                                                        uncertainty_weight=0.0)
        else:
            logger.info(f"############ Scoring predictions (with sampling mode:{sampling_mode}, query function: {acquisition_mode}) over pool set. ############")
            img_unc_dict = compute_inst_surprise_images(predicted_instances, 
                                                        acquisition_mode, 
                                                        NMS=NMS, 
                                                        weights=(weight_cls, weight_reg))
            # aggregate surprise of each detecton to the surprise of one image
            # function that modifies img_unc_dict
            img_unc_dict = aggregate_inst_each_image(aggregate_mode, img_unc_dict)
            # aggregate predicted category information from detected instances

            selected_sim_data_json_path = os.path.join(save_folder, f'selected_sim_data_{save_suffix}.json')
            sampling_hyper_param_dict = {
                "TopN":{},
                "RandomTopNBalancedSyn":{
                                        "selected_sim_data_json_path": selected_sim_data_json_path,
                                        "num_class": len(predicted_instances[0]['cls_prob'])
                                        },
                "RandomTopNBalancedReal":{
                    "num_class": len(predicted_instances[0]['cls_prob'])
                    },
                "Hybrid":{
                    "random_ratio":  (1 - cur_iter*0.1) if cur_iter*0.1<1. else 0.
                    },
            }
            selected_image_id_list = sample_img_based_on_img_unc(sampling_mode, 
                                                                num_select,
                                                                img_unc_dict, 
                                                                sampling_hyper_param_dict, 
                                                                cur_pool_set_instances_gt,
                                                                logger=logger)

    # select images from current pool set
    images = cur_pool_set_instances_gt['images']
    annotations = cur_pool_set_instances_gt['annotations']

    # filter images based on selected_image_id_list
    images = funcy.lfilter(lambda i: int(i['id']) in selected_image_id_list, images)
    if "Coreset" in sampling_mode:
        for image in images:
            image['cls_pen_embs'] = img_feat_dict[image['id']]
    # filter annotations
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    annotations = funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

    # update selected json file
    logger.info(f"############ Selected {len(images)} data points from pool set. ############")
    if prev_selected_data_path is not None:
        logger.info(f"############ ACCUMLATE data from {prev_selected_data_path}! ############")
        selected_instances = json.load(open(prev_selected_data_path, 'r')) 
        logger.info(f"############ Accumulating previously selected {len(selected_instances['images'])} data points. ############")
        selected_instances['images'].extend(images)
        selected_instances['annotations'].extend(annotations)
    else:
        # dealing with the first iteration
        selected_instances = copy.deepcopy(cur_pool_set_instances_gt)
        selected_instances['images'] = images
        selected_instances['annotations'] = annotations
    selected_data_path = os.path.join(save_folder, f'selected_data_{save_suffix}.json')
    logger.info(f"############ There are totally {len(selected_instances['images'])} data points selected. ############")
    with open(selected_data_path, 'w') as fp:
        json.dump(selected_instances, fp, indent=4,separators=(',', ': '))
    logger.info(f"############ Saving to {selected_data_path} ############")

    # update pool set
    new_pool_set_path = os.path.join(save_folder, f'pool_set_{save_suffix}.json')
    save_img_in_list(
        source_json_file_or_ins=cur_pool_set_json, 
        save_json_file=new_pool_set_path, 
        img_id_list=selected_image_id_list, 
        in_list=False, # save images that are in the list of selected_image_id_list
        logger=logger)

    return selected_data_path, new_pool_set_path, selected_sim_data_json_path