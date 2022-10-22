import numpy as np
import funcy 
import json

def bernoulli_ent(prob_list):
    prob_list = np.array(prob_list)
    ent = -prob_list*np.log(prob_list) - (1-prob_list)*np.log(1-prob_list)
    return ent.sum()

def mvg_ent(cov):
    cov = np.array(cov)
    dim = cov.shape[0]
    ent = 0.5*dim*(1 + np.log(2*np.pi)) + 0.5*np.log(np.linalg.det(cov))
    return ent

def imbalance_ratio(hist):
    return np.std(hist)/np.mean(hist)

def class_kldiv_uniform(hist):
    num_cls = len(hist)
    uniform_dist = np.ones(num_cls) / num_cls
    return 1/num_cls * np.sum(np.log(uniform_dist/hist)) + np.log(num_cls)

def count_class_statistic(num_class, cur_pool_set_instances, temp_img_id_list, logger, descrp=""):
    # loop over annoations to calculate class histogram
    cls_hist = np.zeros(num_class)
    for inst in cur_pool_set_instances:
        if inst['image_id'] in temp_img_id_list:
            cls_hist[inst['category_id']] += 1
    # logger.info(f"############ Current existing img_idx list: {temp_img_id_list}. ############")
    logger.info(f"{descrp}")
    logger.info(f"############ Current existing cls_hist: {cls_hist}. ############")
    logger.info(f"############ KLdive between uniform and Class dist.: {class_kldiv_uniform(cls_hist)}. ############")
    
    return cls_hist

def compute_instance_surprise(instance, 
                            mode='cls', 
                            weight_cls=1.0,
                            weight_reg=1.0):
    if mode == 'cls':
        surprise = bernoulli_ent(instance['cls_prob'])
    if mode == 'cls_bald':
        surprise = instance['mutual_info']
    if mode == 'cls_batch_bald':
        surprise = instance['mutual_info']
    elif mode == 'reg':
        surprise = mvg_ent(instance['bbox_covar'])
    elif mode == 'both':
        surprise= weight_cls*bernoulli_ent(instance['cls_prob']) + \
                  weight_reg*mvg_ent(instance['bbox_covar'])
    elif mode == 'max':
        surprise= max(bernoulli_ent(instance['cls_prob']), mvg_ent(instance['bbox_covar']))

    return surprise


def save_img_in_list(source_json_file_or_ins, save_json_file, img_id_list, in_list=True, logger=None):
    if isinstance(source_json_file_or_ins, str):
        source_data_instances = json.load(open(source_json_file_or_ins, 'r'))
    else:
        source_data_instances = source_json_file_or_ins
    images = source_data_instances['images']
    annotations = source_data_instances['annotations']
    # filter images
    if in_list:
        images = funcy.lfilter(lambda i: int(i['id']) in img_id_list, images)
    else:
        images = funcy.lfilter(lambda i: int(i['id']) not in img_id_list, images)
    source_data_instances['images'] = images
    if logger is not None:
        logger.info(f"############ There are totally {len(source_data_instances['images'])} images saved. ############")
    # filter annotations
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    annotations = funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)
    source_data_instances['annotations'] = annotations
    if logger is not None:
        logger.info(f"############ Saving to {save_json_file} ############")
    with open(save_json_file, 'w') as fp:
        json.dump(source_data_instances, fp, indent=4,separators=(',', ': '))
    
    # return save_json_file


def get_oc_uc_image_idx_vec(num_class, cls_hist, img_cls_count_vec):
    """
    return image idx vec with dim #image*1 for over and under represented classes
    in oc, image idx in an ascending order of scores representing the extend how likely the images include over-represented classes 
    in uc, image idx in a descending order of scores representing the extend how likely the images include under-represented classes 
    """
    # get over-represented (oc) and under-represented (uc) cls list (oc_lst, uc_lst) and 
    # their vector saving the extend to the mean cls value    
    over_cls_diff_vec = np.zeros(num_class)
    under_cls_diff_vec = np.zeros(num_class)
    mean_num_cls = np.mean(cls_hist)
    for cat, cat_num in enumerate(cls_hist):
        if cat_num > mean_num_cls:
            over_cls_diff_vec[cat] = np.abs(cls_hist[cat] - mean_num_cls)
        elif cat_num < mean_num_cls:
            under_cls_diff_vec[cat] = np.abs(cls_hist[cat] - mean_num_cls)
    under_cls_diff_vec /= np.sum(under_cls_diff_vec) # np.abs(under_cls_diff_vec - mean_num_cls) / np.sum(np.abs(cls_hist - mean_num_cls))
    over_cls_diff_vec /= np.sum(over_cls_diff_vec) # np.abs(over_cls_diff_vec - mean_num_cls) / np.sum(np.abs(cls_hist - mean_num_cls))
    
    # print(f"under_cls_diff_vec: {under_cls_diff_vec}")
    # print(f"over_cls_diff_vec: {over_cls_diff_vec}")
    # compute overlapping matrix of oc images whose size we want to minize 
    # and sort idx in an ascending manner
    oc_sorted_idx_arr = np.argsort(img_cls_count_vec.dot(over_cls_diff_vec))
    # compute overlapping matrix of uc images whose size we want to maximize 
    # and sort idx in a descending manner
    uc_sorted_idx_arr =  np.argsort(img_cls_count_vec.dot(under_cls_diff_vec))[::-1]
    # print(f"oc_sorted_idx_arr: {oc_sorted_idx_arr}")
    # print(f"uc_sorted_idx_arr: {uc_sorted_idx_arr}")

    return oc_sorted_idx_arr, uc_sorted_idx_arr

def search_images_with_least_oc_most_uc(search_range, oc_sorted_idx_arr, uc_sorted_idx_arr):
    img_idx_overlap_mat = np.tile(oc_sorted_idx_arr[None,:search_range], (search_range, 1)) \
        - np.tile(uc_sorted_idx_arr[:search_range, None], (1, search_range))
    # print(f"oc_sorted_idx_arr[None,:search_range] shape: {oc_sorted_idx_arr[None,:search_range].shape}")
    # print(f"uc_sorted_idx_arr[:search_range, None]: {uc_sorted_idx_arr[:search_range, None].shape}")
    idx_lst_uc, idx_lst_oc = np.where(img_idx_overlap_mat==0)
    # print(f"img_idx_overlap_mat: {img_idx_overlap_mat}")
    # print(f"idx_lst_uc: {idx_lst_uc}")
    # print(f"uc_sorted_idx_arr[idx_lst_uc[:num_selected]]:{uc_sorted_idx_arr[idx_lst_uc[:num_selected]]}")
    num_opt_img_idx = idx_lst_oc.shape[0]

    return idx_lst_uc, num_opt_img_idx

def determinstic_img_cls_balancing_algorithm(num_selected, 
                                            cls_hist, 
                                            img_cls_count_vec,  
                                            logger,
                                            selected_sim_image_id_list,
                                            decay_factor=40,
                                            decay_step=2):
    num_class = cls_hist.shape[0]
    num_img = img_cls_count_vec.shape[0]
    # image class score vec with dim #image*1 for over (minimize) and under (maximize) represented classes
    oc_sorted_idx_arr, uc_sorted_idx_arr = \
        get_oc_uc_image_idx_vec(num_class, cls_hist, img_cls_count_vec)

    # get idxes shared from both arrays which fulfill the minmax optimization above
    cur_selected_img_id_list = []
    while (len(cur_selected_img_id_list) < num_selected) and (decay_factor > 5):
        search_range = int(num_img/decay_factor) # expand range of searching in each iteration
        logger.info(f"Computing overlapping matrix from the first {search_range} sorted images.")
        idx_lst_uc, num_opt_img_idx = search_images_with_least_oc_most_uc(search_range, oc_sorted_idx_arr, uc_sorted_idx_arr)
        logger.info(f"Got {num_opt_img_idx} overlapping images (decay_factor:{decay_factor}) and {num_selected} are required.")
        decay_factor -= decay_step

        # avoid selecting the same image
        temp_list = selected_sim_image_id_list + cur_selected_img_id_list
        for cur_img_idx in uc_sorted_idx_arr[idx_lst_uc[:num_selected]]:
            if cur_img_idx not in temp_list:
                cur_selected_img_id_list.append(cur_img_idx)

    return cur_selected_img_id_list