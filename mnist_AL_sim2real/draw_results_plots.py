import json
import argparse
import os
from xml.dom import NotFoundErr

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

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
        content_dict = l['iter{}'.format(idx)]
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
                title='', 
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
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend_size)    # legend fontsize
    x_step = acq_size*5
    y_step = 2.5
    print("num_runs:{}, num_iter:{}".format(num_run, num_iter))
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'lime', 'gray']
    # sty_list = ['-*r', '--sg', ':db', '-vc', ':xm', '-.^y', '-.3k', '-.|k', '-.hk']
    sty_list = ['-1r', '--2g', ':3b', '-4c', ':xm', '-..y', '-*k', '-.xm','-.k', '-.m']
    if "MNIST" in flag:
        y_sim_mAP = 60
        y_real_mAP = 91
        label_real = "trained on MNIST-M (55k)"
        label_sim = "trained on MNIST (60k)"
        tick_texts = []
        for size in np.arange(0, acq_size*num_iter+3, x_step):
            # tick_texts.append(f"{int(size)}")
            tick_texts.append(f"{(size/550.):.1f}%")

    x = np.arange(0, num_iter*acq_size+1, acq_size) # np.linspace(0, num_iter*acq_size, num=num_iter)
    plt.xlim(0, acq_size*num_iter+3)
    plt.xticks(np.arange(0, acq_size*num_iter+3, x_step), tick_texts)

    # draw result curves
    plt.xlabel("Total queried data size")
    plt.ylabel("Accuracy (%)")
    idx = 0
    y_max, y_min = 0, 100 
    key_order = ["random", "ent_ss0.01", "bbald", "clue_ent", "ent"]
    replace_dict = {"ent_ss0.01": "entropy_ss(proposed)", "ent": "entropy", "clue_ent":"clue", "bbald": "batch_bald"}
    # for _, key in enumerate(al_result_dict.keys()):
    for key in key_order:
        if len(al_result_dict[key]) > 0:
            print("Processing ", key, " idx:", idx)
            arr_mul_run = np.array(al_result_dict[key]) * 100.
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
    ylim_min = min(int(y_sim_mAP-2), int(y_min)-2)
    ylim_max = max(int(y_real_mAP+3), int(y_max)+3)
    plt.ylim(ylim_min, ylim_max)
    plt.yticks(np.arange(ylim_min, ylim_max, y_step))
    y_sim = np.ones_like(x) * y_sim_mAP
    y_real = np.ones_like(x) * y_real_mAP
    plt.plot(x, y_real, sty_list[-1], label=label_real, linewidth=1.0)
    plt.plot(x, y_sim, sty_list[-2], label=label_sim, linewidth=1.0)
    plt.title(title)

    plt.legend()
    if new_path is not None:
        print("saved to ", new_path)
        plt.savefig(new_path)
    else:
        plt.show()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(
        description="a script for drawing figures"
    )
    parser.add_argument("--res-path", type=str, default=None, help="folder that includes json reslut files")
    parser.add_argument("--out-path", type=str, default=None, help="the path that saves output figure")
    args = parser.parse_args()

    path_root = args.res_path # os.getcwd()+"/laaos_results/al_training_2_json/"
    out_path = args.out_path
    json_path_list = os.listdir(path_root)
    
    al_result_dict = {}
    # load results
    for json_path in json_path_list:
        mAP_list = correct_json_reader(os.path.join(path_root, json_path), iter_num=50)

        if mAP_list is None:
            print(f"Coundn't find {json_path} and move to the next.\n")
            raise Exception(f"Coundn't find {json_path} and move to the next.")
            # continue
        else:
            print(f"Read {json_path}")
        legend = os.path.split(json_path)[-1][:-10] # string before "_rnd*.json"
        if legend in al_result_dict.keys():
            al_result_dict[legend].append(mAP_list)
        else:
            al_result_dict[legend] = []
            al_result_dict[legend].append(mAP_list)

    # draw figures
    plot_with_std(al_result_dict, 
                    20, 
                    3, 
                    50, 
                    title="Active learning curves from MNIST to MNIST-M",
                    new_path= out_path, # os.getcwd()+"/laaos_results/results_figs/acq20_lr1e-5_1e-3_warmup.pdf", 
                    flag="MNIST2MNISTM",
                    legend_size=13)
