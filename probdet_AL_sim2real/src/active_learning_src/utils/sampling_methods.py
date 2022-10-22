import numpy as np
import json
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from detectron2.data.catalog import MetadataCatalog

from src.active_learning_src.utils.batch_bald.batch_bald_utils import get_batchbald_batch
from src.active_learning_src.utils.utils import (
    count_class_statistic,
    imbalance_ratio, 
    determinstic_img_cls_balancing_algorithm)

use_cuda = torch.cuda.is_available()

print(f"use_cuda: {use_cuda}")

device = "cuda" if use_cuda else "cpu"

def sample_img_batch_bald(predicted_instances, num_select, num_samples=1000):
    # num_inst_each_img = len(predicted_instances[0])
    for idx, instance in enumerate(predicted_instances):
        if idx == 0:
            pool_inst_probs_N_K_C = instance.pred_cls_probs_samples
            img_idx_N = instance.image_id
        else:
            pool_inst_probs_N_K_C = torch.cat((pool_inst_probs_N_K_C, instance.pred_cls_probs_samples), dim=0)
            img_idx_N = torch.cat((img_idx_N, instance.image_id), dim=0)

    selected_instances_id_list, scores_list = get_batchbald_batch(pool_inst_probs_N_K_C, 
                                                                num_select, 
                                                                num_samples=num_samples,
                                                                img_idx_N=img_idx_N,
                                                                dtype=torch.double,
                                                                device=device)
    selected_image_id_list = []
    for inst_id in selected_instances_id_list:
        selected_image_id_list.append(img_idx_N[inst_id])
    return selected_image_id_list

def sample_img_clue(predicted_instances, num_select, acquisition_mode):
    for idx, instance in enumerate(predicted_instances):
        if idx == 0:
            pen_embs = np.array(instance['cls_pen_embs'])[None, :]
            instance_img_idx = [instance['image_id']]
            sample_weights = [instance["acqusition_dict"][acquisition_mode]]
        else:
            pen_embs = np.concatenate((pen_embs, np.array(instance['cls_pen_embs'])[None, :]), axis=0)
            instance_img_idx = np.concatenate((instance_img_idx, [instance['image_id']]), axis=0)
            sample_weights = np.concatenate((sample_weights, [instance["acqusition_dict"][acquisition_mode]]), axis=0)

    selected_image_id_list = Clue_sampling(num_select, pen_embs, sample_weights, instance_img_idx)
    return selected_image_id_list

def sample_img_coreset(predicted_instances, 
                    num_select, 
                    prev_selected_data_path,
                    uncertainty_weight=0.0) :
    
    img_feat_dict = {}
    if prev_selected_data_path is None:
        # selected_image_id_list = TopN_sampling(img_unc_dict, num_select)
        image_id_list = list(set([inst['image_id'] for inst in predicted_instances]))
        selected_image_id_list = np.random.choice(image_id_list, size=num_select, replace=False)
        selected_image_id_list = list(selected_image_id_list)
        for inst in predicted_instances:
            if inst['image_id'] in selected_image_id_list:
                if inst['image_id'] in img_feat_dict.keys():
                    img_feat_dict[inst['image_id']].append(inst['cls_pen_embs'])
                else:
                    img_feat_dict[inst['image_id']] = [inst['cls_pen_embs']]
    else:
        selected_instances = json.load(open(prev_selected_data_path, 'r')) 
        prev_inst_feat = []
        inst_img_id_list = []
        inst_feat_list = []
        for inst in predicted_instances:
            inst_img_id_list.append(inst['image_id'])
            inst_feat_list.append(inst['cls_pen_embs'])

        for img in selected_instances['images']:
            prev_inst_feat.extend(img['cls_pen_embs'])
        selected_image_id_list = Coreset_sampling(inst_feat_list, 
                                                inst_img_id_list,
                                                prev_inst_feat, 
                                                num_select,
                                                dist_fnc="euclidean") # dist_fnc= {"euclidean", "cosine-sim"}
        for inst in predicted_instances:
            if inst['image_id'] in selected_image_id_list:
                if inst['image_id'] in img_feat_dict.keys():
                    img_feat_dict[inst['image_id']].append(inst['cls_pen_embs'])
                else:
                    img_feat_dict[inst['image_id']] = [inst['cls_pen_embs']]
                    
    return selected_image_id_list, img_feat_dict

def Clue_sampling(num_select, pen_embs, sample_weights, instance_img_idx):
    # Run weighted K-means over embeddings
    km = KMeans(num_select)
    km.fit(pen_embs, sample_weight=sample_weights)

    # Find nearest neighbors to inferred centroids
    dists = euclidean_distances(km.cluster_centers_, pen_embs)
    sort_idxs = dists.argsort(axis=1)
    img_idxs = []
    ax, rem = 0, num_select
    while rem > 0:
        inst_idx_list = list(sort_idxs[:, ax][:rem])
        for inst_idx in inst_idx_list:
            img_idxs.append(instance_img_idx[inst_idx])
        img_idxs = list(set(img_idxs))
        rem = num_select - len(img_idxs)
        ax += 1

    return img_idxs

def TopN_sampling(img_id_score_dict, num_selection):
    # rank the image id based on the surprise in a descending manner
    image_id_rank_list = sorted(img_id_score_dict, key=img_id_score_dict.get, reverse=True)
    selected_image_id_list = image_id_rank_list[:num_selection]
    return selected_image_id_list

def Balanced_selection(num_to_select_buffer, cls_hist, img_cls_count_mat, logger, selected_image_id_list=[]):
    """
    function to search images with specific object categories, 
    in order to balance the class distribution of the dataset
    Args:
        num_to_select_buffer: int, maximal number of images to be selected;
        cls_hist: np.array, initial class histogram of the current dataset;
        img_cls_count_mat: np.array, dim: num_img*num_cls, first dim includes image idx, second dim include #cat in this image;
        selected_image_id_list: previous selected image id, if equal to [], no need to consider;

    Return:
        selected_image_id_list: list, selected image idx
    """

    # initialize placeholder for to-be-selected images
    num_iter = min(10, num_to_select_buffer) # int(num_to_select_buffer/2)
    for iter in range(1, num_iter+1):
        num_to_select = int(num_to_select_buffer/num_iter)
        logger.info(f"##### Selecting {num_to_select} synthetic images in iter{iter}/{num_iter}. #####")
        cur_selected_img_id_list = determinstic_img_cls_balancing_algorithm(num_to_select, 
                                                                        cls_hist, 
                                                                        img_cls_count_mat, 
                                                                        logger,
                                                                        selected_image_id_list, 
                                                                        decay_factor=40,
                                                                        decay_step=2)
        logger.info(f"Get the first {num_to_select} images in underrepsented array, which are {cur_selected_img_id_list}.")

        # update cls_hist
        for img_idx in cur_selected_img_id_list:
            # print(f"img_cls_count_mat[img_idx, :]: {img_cls_count_mat[img_idx, :]}")
            for cat_idx, cat_num in enumerate(img_cls_count_mat[img_idx, :]):
                if cat_num > 0:
                    cls_hist[cat_idx] += 1
        
        # check imbalance ratio
        logger.info(f"After incorporating images in iter{iter}: {cls_hist}")
        logger.info("The imbalance_ratio is {:.3f}".format(imbalance_ratio(cls_hist)))
        if imbalance_ratio(cls_hist) < 1e-4:
            break
        selected_image_id_list.extend(cur_selected_img_id_list)

    return selected_image_id_list

def BalancedReal_sampling(selected_image_id_list, cls_hist, to_be_selected_instances, logger):
    # relevant params
    num_class = cls_hist.shape[0]
    num_selected = len(selected_image_id_list)
    num_to_select_buffer = num_selected
    logger.info("Before incorporating real images: imbalance_ratio is {:.3f}".format(imbalance_ratio(cls_hist)))
    logger.info(f"{cls_hist}")

    # obtain info from to-select dataset
    num_img = len(to_be_selected_instances['images'])
    img_cls_count_mat = np.zeros((num_img, num_class))
    img_id2continuous_idx_dict = {}
    for idx, image in enumerate(to_be_selected_instances['images']):
        img_id2continuous_idx_dict[image['id']] = idx
    for ann in to_be_selected_instances['annotations']:
        img_cls_count_mat[img_id2continuous_idx_dict[ann['image_id']], int(ann['category_id'])-1] += 1
        
    # selected_image_id_list gets updated in the function
    selected_image_id_list = Balanced_selection(num_to_select_buffer, 
                                                cls_hist, 
                                                img_cls_count_mat, 
                                                logger, 
                                                selected_image_id_list=selected_image_id_list)
    
    return selected_image_id_list

def BalancedSyn_sampling(num_selected, 
                        cls_hist, 
                        logger=None, 
                        sim_dataset_name='ycbv_sim'):

    # relevant params
    num_class = cls_hist.shape[0]
    num_to_select_buffer = num_selected
    logger.info("Before incorporating sim images: imbalance_ratio is {:.3f}".format(imbalance_ratio(cls_hist)))
    logger.info(f"{cls_hist}")

    # obtain info from to-select dataset
    sim_json_file = MetadataCatalog.get(sim_dataset_name).json_file
    to_be_selected_sim_data_instances = json.load(open(sim_json_file, 'r'))
    num_img = len(to_be_selected_sim_data_instances['images'])
    img_cls_count_mat = np.zeros((num_img, num_class))
    for ann in to_be_selected_sim_data_instances['annotations']:
        img_cls_count_mat[int(ann['image_id'])-1, int(ann['category_id'])-1] += 1

    selected_sim_image_id_list = Balanced_selection(num_to_select_buffer, cls_hist, img_cls_count_mat, logger)
    
    return selected_sim_image_id_list

def Coreset_sampling(inst_feat_list, 
                    inst_img_id_list,
                    prev_inst_feat_list, 
                    num_select,
                    dist_fnc="euclidean"): 
    # select num_select samples based on algorithm 1 in https://arxiv.org/pdf/1708.00489.pdf
    # construct similarity matrix between pool_img_feat_dict and selected_img_dict
    feat_dim = len(inst_feat_list[0])
    len_pool = len(inst_img_id_list)
    len_labeled = len(prev_inst_feat_list)

    pool_inst_feat_arr = np.asarray(inst_feat_list).reshape(len_pool, feat_dim) # len_pool*feat_dim
    prev_inst_feat_arr = np.asarray(prev_inst_feat_list).reshape(len_labeled, feat_dim) # len_labeled*feat_dim

    if dist_fnc == "euclidean":
        dist_mat_cls = np.tile(pool_inst_feat_arr[:,None,:], (1,len_labeled,1)) - np.tile(prev_inst_feat_arr[None, :, :], (len_pool,1,1)) # (len_pool, len_labeled, cls_dim)
        dist_mat = np.linalg.norm(dist_mat_cls, axis=2) # (len_pool, len_labeled)
        dist_mat = dist_mat 
    elif dist_fnc == "cosine-sim":
        in_prod = pool_inst_feat_arr.dot(prev_inst_feat_arr.T)  # (len_pool, len_labeled)
        pool_norm_arr = np.linalg.norm(pool_inst_feat_arr, axis=1, keepdims=True)
        labeled_norm_arr = np.linalg.norm(prev_inst_feat_arr, axis=1, keepdims=True).T
        dist_mat = -(in_prod/pool_norm_arr.dot(labeled_norm_arr))
        dist_mat = np.exp(dist_mat)
    
    # adding effects from unc of each image, distance is propotional to uncertainty
    # to obtain the N data points with **largest** **minimum** (maxmin) distance to the selected dataset
    # intuition behind this is to get the **most different (informative)** data points that are
    # close to the boundary of the previously selected dataset
    nearest_neigh_mat = np.amin(dist_mat, axis=1, keepdims=True)
    ordered_nn_idx_lst = np.argsort(nearest_neigh_mat, axis=0)
    inst_counter = 0
    selected_image_id_list = []
    while len(selected_image_id_list) < num_select:
        inst_counter += 1
        select_inst_idx = ordered_nn_idx_lst[-inst_counter][0]
        if inst_img_id_list[select_inst_idx] not in selected_image_id_list:
            selected_image_id_list.append(inst_img_id_list[select_inst_idx])
    
    return selected_image_id_list


def Coreset_sampling_adapted(pool_img_feat_dict, 
                            selected_img_feat_lst, 
                            pool_img_unc_dict, 
                            num_select, 
                            dist_fnc="euclidean", 
                            unc_cof=0.01):
    # select num_select samples based on algorithm 1 in https://arxiv.org/pdf/1708.00489.pdf
    # construct similarity matrix between pool_img_feat_dict and selected_img_dict
    feat_dim = len(selected_img_feat_lst[0])
    len_pool = len(pool_img_feat_dict.keys())
    len_labeled = len(selected_img_feat_lst)

    img_index_lst = list(pool_img_feat_dict.keys())
    pool_img_unc_arr = np.array(list(pool_img_unc_dict.values())).reshape(len_pool, 1)
    img_feat_arr = np.array(list(pool_img_feat_dict.values())).reshape(len_pool, feat_dim)
    selected_img_feat_arr = np.array(selected_img_feat_lst).reshape(len_labeled, feat_dim)
    if dist_fnc == "euclidean":
        dist_mat_cls = np.tile(img_feat_arr[:,None,:], (1,len_labeled,1)) - np.tile(selected_img_feat_arr, (len_pool,1,1)) # (len_pool, len_labeled, cls_dim)
        dist_mat = np.linalg.norm(dist_mat_cls, axis=2) # (len_pool, len_labeled)
        dist_mat = dist_mat + unc_cof*pool_img_unc_arr
    elif dist_fnc == "cosine-sim":
        in_prod = img_feat_arr.dot(selected_img_feat_arr.T)  # (len_pool, len_labeled)
        pool_norm_arr = np.linalg.norm(img_feat_arr, axis=1, keepdims=True)
        labeled_norm_arr = np.linalg.norm(selected_img_feat_arr, axis=1, keepdims=True).T
        dist_mat = -(in_prod/pool_norm_arr.dot(labeled_norm_arr))
        dist_mat = np.exp(dist_mat) * pool_img_unc_arr
    
    # adding effects from unc of each image, distance is propotional to uncertainty
    # to obtain the N data points with **largest** **minimum** (maxmin) distance to the selected dataset
    # intuition behind this is to get the **most different (informative)** data points that are
    # close to the boundary of the previously selected dataset
    nearest_neigh_mat = np.amin(dist_mat, axis=1, keepdims=True)
    ordered_nn_idx_lst = np.argsort(nearest_neigh_mat, axis=0)
    selected_idx_arr = ordered_nn_idx_lst[-num_select:] # in an ascending order and select the topN largest ones
    selected_image_id_list = []
    for id in selected_idx_arr.squeeze().tolist():
        selected_image_id_list.append(img_index_lst[id])
    
    return selected_image_id_list

def BalancedSoftmaxFeat_sampling(img_id_score_dict, num_selection):
    pass
