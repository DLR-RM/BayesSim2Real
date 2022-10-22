import json
import os
import argparse
from xml.dom import NotFoundErr

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

def incorrect_json_reader(json_path):
    with open(json_path) as f:
        contents = f.readlines()

    mAP_lst = []
    for idx, l in enumerate(contents):
        if "\"bbox/AP\"" in l:
            mAP_lst.append(float(contents[idx+1].strip(" ")[:6]))

    return mAP_lst

def correct_json_reader(json_path, iter_num=10):
    try:
        with open(json_path) as f:
            contents = json.load(f)
    except IOError as e:
        print("\nThere is an error {}.\n".format(e))
        return None
    mAP_lst = []
    for idx, l in enumerate(contents[:iter_num]):
        # print(contents)
        content_dict = l['iter{}'.format(idx+1)]
        if "bbox/AP" in content_dict.keys():
            if isinstance(content_dict["bbox/AP"], list):
                mAP = content_dict["bbox/AP"][0]
            else:
                mAP = content_dict["bbox/AP"]
        else:
            mAP = content_dict["AP"]
        mAP_lst.append(float(mAP))

    return mAP_lst

def plot_with_std(al_result_dict, 
                acq_size, 
                num_run, 
                num_iter, 
                exp_name='', 
                title="",
                new_path=None, 
                flag="edan",
                legend_size=15):
    # fig = plt.figure()
    # ax = plt.axes()
    # u_idx = exp_name.rindex("_")
    # exp_name = exp_name[:u_idx]
    plt.figure()
    # plt.rcParams.update({'font.size': 13.3})
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend_size)    # legend fontsize
    print("num_runs:{}, num_iter:{}".format(num_run, num_iter))
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'gray']
    # sty_list = ['-*r', '--sg', ':db', '-vc', ':xm', '-.^y', '-.3k', '-.|k', '-.hk']
    sty_list = ['-1r', '--2g', ':3b', '-4c', ':xm', '-..y', '-*k', '-.xm','-.k', '-.m']
    y_lim_step = 2.5
    if "edan" in flag:
        if "faster-rcnn" in flag:
            y_sim_mAP = 54.5
            y_real_mAP = 86.0
        else:
            y_sim_mAP = 45.2
            y_real_mAP = 82.0
        label_real = "real only (~0.5k)"
        label_sim = "sim only (~10k)"
        tick_texts = []
        for size in np.arange(0, acq_size*num_iter+3, acq_size):
            tick_texts.append(f"{int(size/5)}%")
    elif "ycbv" in flag:
        if "faster-rcnn" in flag:
            y_sim_mAP = 49.6
            y_real_mAP = 67.8
        else:
            y_sim_mAP = 39.8
            y_real_mAP = 67.8 
        label_real = "real only (~1.5k)"
        label_sim = "sim only (~50k)"
        tick_texts = []
        for size in np.arange(0, acq_size*num_iter+3, acq_size):
            tick_texts.append(f"{int(size/14.4)}%")
    elif "sam" in flag:
        ylim_min = 55.
        y_sim_mAP = ylim_min + 0.3
        y_lim_step = 2.5
        y_real_mAP = 91.0 * 0.95
        label_real = "real only (~2k)"
        label_sim = "sim only (~2.3k)"
        tick_texts = []
        for size in np.arange(0, acq_size*num_iter+3, acq_size):
            tick_texts.append(f"{int(size/20)}%")

    x = np.arange(0, num_iter*acq_size+1, acq_size) # np.linspace(0, num_iter*acq_size, num=num_iter)
    plt.xlim(0, acq_size*num_iter+3)
    plt.xticks(np.arange(0, acq_size*num_iter+3, acq_size), tick_texts)

    # draw result curves
    plt.xlabel("Total queried data size")
    plt.ylabel("mAP (%)")
    idx = 0
    y_max, y_min = 0, 100 
    replace_dict = {"both_avg_ss": "both_avg_ss(proposed)", "both_max_ss": "both_max_ss(proposed)"}
    for _, key in enumerate(al_result_dict.keys()):
        if len(al_result_dict[key]) > 0:
            print("Processing ", key, " idx:", idx)
            arr_mul_run = np.array(al_result_dict[key])
            if np.max(arr_mul_run) > y_max:
                y_max = np.max(arr_mul_run)
            if np.min(arr_mul_run) < y_min:
                y_min = np.min(arr_mul_run)
            mean_arr = np.mean(arr_mul_run, axis=0)
            std_arr = np.std(arr_mul_run, axis=0)
            if key in replace_dict.keys(): key = replace_dict[key]
            label = key + f"({np.mean(mean_arr):.2f}%)"
            print(f"mean:{np.mean(mean_arr)} -+ std:{np.mean(std_arr)}")
            dy = np.concatenate((np.array([0]), std_arr), axis=0)
            y = np.concatenate((np.array([y_sim_mAP]), mean_arr), axis=0)

            plt.plot(x, y, sty_list[idx], linewidth=0.3)
            plt.errorbar(x, y, yerr=dy, fmt='', color=color_list[idx], elinewidth=1, capsize=0.7, label=label)
            idx += 1

    # set up axis and ticks
    ylim_min = min(int(y_sim_mAP) if y_sim_mAP != 0. else 0., int(y_min)-2)
    ylim_max = max(int(y_real_mAP+3), int(y_max)+3)
    plt.ylim(ylim_min, ylim_max)
    y_ticks = [0., "...", ]
    for idx, y_tick in enumerate(np.arange(ylim_min, ylim_max, y_lim_step)):
        if idx > 1:
            y_ticks.append(y_tick)
    plt.yticks(np.arange(ylim_min, ylim_max, y_lim_step), y_ticks)
    y_sim = np.ones_like(x) * y_sim_mAP
    y_real = np.ones_like(x) * y_real_mAP
    plt.plot(x, y_real, sty_list[-1], label=label_real, linewidth=1.0)
    plt.plot(x, y_sim, sty_list[-2], label=label_sim, linewidth=1.0)
    # plt.title("{} on daily object dataset for {} random runs".format(exp_name ,num_run))
    title = f"Active learning curves on {flag} data set"
    plt.title(title)

    plt.legend()
    if new_path is not None:
        print("saved to ", new_path)
        plt.savefig(new_path)
    else:
        plt.show()


def get_exp_folder_name(acq_size, 
                        iter_num, 
                        acq_func, 
                        agg_mode, 
                        sampling_mode, 
                        rnd, 
                        eval_iter_early_stopping, 
                        max_ep,
                        lr, 
                        random=False,
                        suffix=None, 
                        bbald_num_samples=1000):
    if random: 
            folder_name = (f"acq{acq_size}_iter{iter_num}"
                f"_rnd{rnd}_eval{eval_iter_early_stopping}_lr{str(lr)}_RANDOM")
            legend = "random"
    else:
        if acq_func == "cls_batch_bald" and "RandomTopN" not in sampling_mode:
            folder_name = (f"acq{acq_size}_iter{iter_num}"
                f"_{acq_func}_{bbald_num_samples}"
                f"_rnd{rnd}_eval{eval_iter_early_stopping}_lr{str(lr)}")
            legend = f"{acq_func}"
        else:
            if "clue" in sampling_mode:
                folder_name = (f"acq{acq_size}_iter{iter_num}"
                    f"_{acq_func}_{sampling_mode}"
                    f"_rnd{rnd}_eval{eval_iter_early_stopping}_lr{str(lr)}")
                legend = f"{acq_func}_{sampling_mode}"
                    
            elif "Corset" in sampling_mode:
                folder_name = (f"acq{acq_size}_iter{iter_num}"
                    f"_{sampling_mode}"
                    f"_rnd{rnd}_eval{eval_iter_early_stopping}_lr{str(lr)}")
                legend = f"cls_coreset"
            else:
                folder_name = (f"acq{acq_size}_iter{iter_num}"
                    f"_{agg_mode}_{acq_func}_{sampling_mode}"
                    f"_rnd{rnd}_eval{eval_iter_early_stopping}_lr{str(lr)}")
                if sampling_mode == "RandomTopN":
                    sampling_mode = "ss"
                legend = f"{acq_func}_{agg_mode}_{sampling_mode}"

    if eval_iter_early_stopping == "0":
        folder_name = folder_name.replace(f"eval{eval_iter_early_stopping}", f"ep{max_ep}")
    
    if suffix == "None":
        suffix = ""
        
    return folder_name+suffix, legend+suffix if "WCls" not in suffix else legend


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--base_path_list",
        nargs="+",
        help="root path for saving the results of active learning.")
    arg_parser.add_argument(
        "--num_iter",
        type=int,
        default=5,
        help="number of loops for active learning.")
    arg_parser.add_argument(
        "--num_select",
        nargs="+",
        default=10,
        help="size of acquisition in each iteration.")
    arg_parser.add_argument(
        "--aggregate_mode",
        nargs="+",
        default="avg",
        help="way to combine scores of each detection into that of one image, ['sum', 'avg']")
    arg_parser.add_argument(
        "--acquisition_mode",
        nargs="+",
        default="cls",
        help="acquisition function, ['cls', 'reg', 'both']")
    arg_parser.add_argument(
        "--sampling_mode",
        nargs="+",
        default="TopN",
        help="sampling strategy, [ 'TopN', 'Coreset', 'RandomTopN', 'TopNBalancedSyn', 'TopNBalancedReal', 'RandomTopNBalancedSyn', 'RandomTopNBalancedReal']")
    arg_parser.add_argument(
        "--random",
        action="store_true",
        help="whether to use random acquisition function")
    arg_parser.add_argument(
        "--rnd_list",
        nargs="+",
        default=[0],
        help="a list of random seed to be used for all scientific computing libraries")
    arg_parser.add_argument(
        "--nms",
        action="store_true",
        help="to use nms or not")
    arg_parser.add_argument(
        "--weight_cls",
        type=float,
        default=1.0,
        help="weight of cls in 'both' acquisition function")
    arg_parser.add_argument(
        "--weight_reg",
        type=float,
        default=1.0,
        help="weight of reg in 'both' acquisition function")
    arg_parser.add_argument(
        "--lr_list",
        nargs="+",
        default="1e-3",
        help="learning rate used for AL")
    arg_parser.add_argument(
        "--early_stopping_list",
        nargs="+",
        default="2",
        help="number of iterations to check for early stopping in AL")
    arg_parser.add_argument(
        "--max_epoch_list",
        nargs="+",
        default="50",
        help="number of iterations to check for early stopping in AL")
    arg_parser.add_argument(
        "--legend_size",
        type=int,
        default=15,
        help="number of iterations to check for early stopping in AL")
    arg_parser.add_argument(
        "--save_fig_name",
        type=str,
        default="")
    # arg_parser.add_argument(
    #     "--fig_title",
    #     type=str,
    #     default="")
    arg_parser.add_argument(
        "--exclude_list",
        nargs="+",
        default=[''],
        help="combinations should be ignored")
    arg_parser.add_argument(
        "--exp_folder_suffix_list",
        nargs="+",
        default=[0],
        help="a list of random seed to be used for all scientific computing libraries")

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    result_file_name = "al_results.json" # "al_results.json", al_results_ycbv_real_test_all
    acq_size_list = args.num_select
    exclude_list = args.exclude_list
    exp_folder_suffix_list = args.exp_folder_suffix_list + [""]
    iter_num = args.num_iter
    acq_func_list = args.acquisition_mode # "cls", "reg", "both", ""
    agg_mode_list = args.aggregate_mode # "avg", "sum"
    sampling_mode_list = args.sampling_mode # "avg", "sum"
    random = args.random
    rnd_list = args.rnd_list
    lr_list = args.lr_list
    eval_early_stopping_list = args.early_stopping_list
    max_ep_list = args.max_epoch_list
    nms = args.nms
    save_fig_name = args.save_fig_name
    num_run, num_iter = len(rnd_list), iter_num
    al_result_dict = {'random': []}
    if "cls_batch_bald" in acq_func_list:
        bbald_num_sample_list = [1000] # , 5000, 10000]
    else:
        bbald_num_sample_list = [0]

    if random:
        random_list = [True]
    else:
        random_list = [True, False]
    for exp_base_path in args.base_path_list: 
        for random_flag in random_list:
            for lr in lr_list:
                for early_stopping in eval_early_stopping_list:
                    for acq_size in acq_size_list:
                        for max_ep in max_ep_list:
                            for sampling_mode in sampling_mode_list:
                                for acq_func in acq_func_list:
                                    for agg_mode in agg_mode_list:
                                        for rnd in rnd_list:
                                            for bbald_num_sample in bbald_num_sample_list:
                                                for suffix in exp_folder_suffix_list: 
                                                    folder_name, legend = \
                                                        get_exp_folder_name(int(acq_size), 
                                                                            iter_num, 
                                                                            acq_func, 
                                                                            agg_mode, 
                                                                            sampling_mode,
                                                                            rnd, 
                                                                            random=random_flag,
                                                                            eval_iter_early_stopping=early_stopping, 
                                                                            max_ep=max_ep,
                                                                            lr=float(lr), 
                                                                            bbald_num_samples=bbald_num_sample if bbald_num_sample !=0 else None,
                                                                            suffix=suffix if suffix !=0 else None)
                                                    for ex_name in exclude_list:
                                                        if ex_name != "" and ex_name in folder_name:
                                                            print(f"SKIPPING {folder_name}")
                                                            folder_name = None
                                                            break
                                                        
                                                    if folder_name is not None:
                                                        json_path = os.path.join(exp_base_path, folder_name, result_file_name)
                                                        mAP_list = correct_json_reader(json_path, iter_num=iter_num)
                                                        if mAP_list is None:
                                                            # print(f"Coundn't find {json_path} and move to the next.\n")
                                                            # raise Exception(f"Coundn't find {json_path} and move to the next.")
                                                            continue
                                                        else:
                                                            print(f"Read {json_path}")
                                                            
                                                        # quick fix for mAP list with length smaller than 10
                                                        if len(mAP_list) < 10:
                                                            for i in range(10-len(mAP_list)):
                                                                mAP_list.append(mAP_list[-1])

                                                        if legend in al_result_dict.keys():
                                                            al_result_dict[legend].append(mAP_list)
                                                        else:
                                                            al_result_dict[legend] = []
                                                            al_result_dict[legend].append(mAP_list)
                                                        print("mAP List of", legend)
                                                        print(["{}: {:.2f}".format(iter+1, r) for iter, r in enumerate(mAP_list)])

    # DRAW PLOTS
    if "edan" in args.base_path_list[0]:
        new_path = f"./object_detection_exp/results/EDAN_{save_fig_name}.pdf"
        flag = "edan"
    elif "ycbv" in args.base_path_list[0]:
        new_path = f"./object_detection_exp/results/YCBV_{save_fig_name}.pdf"
        flag = "ycbv"
    elif "sam" in args.base_path_list[0]:
        new_path = f"./object_detection_exp/results/SAM_{save_fig_name}.pdf"
        flag = "sam"
    if "faster-rcnn" in args.base_path_list[0]:
        flag += "_faster-rcnn"
    plot_with_std(al_result_dict, 
                    int(acq_size), 
                    num_run, 
                    num_iter, 
                    # title=args.fig_title,
                    exp_name=folder_name, 
                    new_path=new_path, 
                    flag=flag,
                    legend_size=args.legend_size)