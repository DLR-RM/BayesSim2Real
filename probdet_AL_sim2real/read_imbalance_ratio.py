import json
import os
import argparse
from posixpath import join

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

def plot_with_std(al_result_dict, acq_size, exp_name='', new_path=None):
    # fig = plt.figure()
    # ax = plt.axes()
    u_idx = exp_name.rindex("_")
    exp_name = exp_name[:u_idx]
    plt.figure()
    num_run, num_iter = len(al_result_dict['random']), len(al_result_dict['random'][0])
    print("num_runs:{}, num_iter:{}".format(num_run, num_iter))
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    line_sty_list = ['-', '--', '-.', ':']
    sty_list = ['-*r', '--sg', ':db', '--c', '--m']
    err_sty_list = ['.r', '.g', '.b']
    plt.xlim(0, acq_size*num_iter+3)
    plt.ylim(40, 100)
    plt.title("{} on daily object dataset for {} random runs".format(exp_name ,num_run))
    plt.xlabel("Total queried data size")
    plt.ylabel("mAP")
    x = np.arange(0, num_iter*acq_size+1, acq_size) # np.linspace(0, num_iter*acq_size, num=num_iter)
    y_sim = np.ones_like(x) * 45.2
    y_real = np.ones_like(x) * 82.0
    plt.plot(x, y_real, sty_list[3], label="all real only (~0.5k)")
    plt.plot(x, y_sim, sty_list[4], label="sim only")
    for idx, key in enumerate(al_result_dict.keys()):
        print("Processing ", key)
        label = key
        arr_mul_run = np.array(al_result_dict[key])
        mean_arr = np.mean(arr_mul_run, axis=0)
        std_arr = np.std(arr_mul_run, axis=0)
        dy = np.concatenate((np.array([0]), std_arr), axis=0)
        y = np.concatenate((np.array([45.2]), mean_arr), axis=0)

        plt.plot(x, y, sty_list[idx])
        plt.errorbar(x, y, yerr=dy, fmt='', color=color_list[idx], elinewidth=1, capsize=1.5, label=label)
    plt.legend()
    if new_path is not None:
        print("saved to ", new_path)
        plt.savefig(new_path)
    else:
        plt.show()

def cls_hist_json_reader(json_path, num_cls):
    cls_hist = np.zeros(num_cls)
    num_anno = 0
    try:
        with open(json_path) as f:
            contents = json.load(f)
    except IOError as e:
        print("There is an error: {}.".format(e))
        return None
    for _, anno in enumerate(contents['annotations']):
        cat = anno['category_id']
        cls_hist[cat-1] += 1
        num_anno += 1

    return cls_hist, num_anno

def kl_div_uniform(cls_dist):
    num_cls = len(cls_dist)
    uniform_dist = np.ones(num_cls)/num_cls
    cls_dist = np.asarray(cls_dist) + 1e-5 # prevent under flow
    kl_div = 0
    for idx, cls in enumerate(cls_dist):
        kl_div += uniform_dist[idx] * -np.log(cls)
    
    return kl_div

def print_imbalance_ratio(folder_name, pre_select_folder_name, iter_num, num_cls):
    imbr_dict = {}
    ir_list = []
    div_list = []
    for iter in range(iter_num):
        if iter == 0:
            json_path = os.path.join(folder_name, f"{pre_select_folder_name}/selected_json_files/selected_data_iter{iter+1}.json")
        else:
            json_path = os.path.join(folder_name, f"iter{iter}/inference/selected_json_files/selected_data_iter{iter+1}.json")
        cls_hist, num_anno = cls_hist_json_reader(json_path, num_cls)
        if cls_hist is not None:
            # print(f"iter{iter}: {:.2f}".format(iter, np.std(cls_hist)/np.mean(cls_hist)))
            # imbr_dict[f"iter{iter}"] = f"{np.std(cls_hist)/np.mean(cls_hist):.2f}"
            # ir_list.append(np.std(cls_hist)/np.mean(cls_hist))
            imbr_dict[f"iter{iter}_ir"] = np.std(cls_hist)*num_cls
            imbr_dict[f'iter{iter}_kl_div_uniform'] = kl_div_uniform(cls_hist/num_anno)
            ir_list.append(imbr_dict[f"iter{iter}_ir"])
            div_list.append(imbr_dict[f'iter{iter}_kl_div_uniform'])
        else: 
            continue
    imbr_dict['mean_ir'] = np.mean(ir_list)
    imbr_dict['kl_div_uniform'] = np.mean(div_list)
    # print(imbr_dict)
    return imbr_dict


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="root path for saving the results of active learning.")
    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    # exp_base_path = args.base_path 
    exp_base_path = "./object_detection_exp/ycbv/retinanet/retinanet_R_50_FPN_3x_ycbv_sim_dropout01_lr1e-4/"

    # have to replace "rnd*" with "rnd"!!!
    # EDAN
    # folder_name_list = [
    #     "acq20_iter10_avg_both_TopN_rnd1_eval2000_lr0.0001_WCls1.0_WReg0.01",
    #     "acq20_iter10_avg_both_RandomTopN_rnd_eval2000_lr0.0001_WCls1.0_WReg0.01",
    #     "acq20_iter10_cls_batch_bald_1000_rnd_eval2000_lr0.0001",
    #     "acq20_iter10_cls_clue_rnd_eval2000_lr0.0001",
    #     "acq20_iter10_Corset_rnd_eval2000_lr0.0001",
    #     "acq20_iter10_rnd_eval2000_lr0.0001_RANDOM"
    #     ]
    # no_loop_file_idx = [0] # index of file name with just on random seed

    # YCBV
    folder_name_list = [
        "acq50_iter10_Corset_rnd1_eval1000_lr0.001",
        ]

    no_loop_file_idx = [0, 1, 2, 3] # index of file name with just one random seed
    imbalance_ratio_dict = {}
    pre_select_folder_name = "pretrained_selected" # "pretrained_selected", "inference"
    
    iter_num = 10
    num_cls = 21 # 5 

    for f_idx, folder_name in enumerate(folder_name_list):
        # imbalance_ratio_dict.update({folder_name: []})
        print(f"Imbalance_ratio of {folder_name}:")
        if f_idx in no_loop_file_idx: 
            rnd_list = [1]
        else: 
            rnd_list = [1, 2, 3]
            
        for rnd in rnd_list:
            rnd = int(rnd) 
            if f_idx in no_loop_file_idx:
                full_folder_name = folder_name  
            else:
                full_folder_name = folder_name.replace("rnd", f"rnd{rnd}")

            full_folder_name = os.path.join(exp_base_path, full_folder_name)
            imbr_dict = print_imbalance_ratio(full_folder_name, pre_select_folder_name, iter_num, num_cls)
            if rnd > 1:
                for key in imbr_dict.keys():
                    imbalance_ratio_dict[folder_name][key] += imbr_dict[key]
            else:
                imbalance_ratio_dict[folder_name] = imbr_dict

        for key in imbr_dict.keys():
            imbalance_ratio_dict[folder_name][key] /= len(rnd_list)
        print(imbalance_ratio_dict[folder_name])
        print('\n')
