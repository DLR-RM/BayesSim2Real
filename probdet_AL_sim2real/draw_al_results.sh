#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RUN_CODE="tools/read_json_al_results.py"

PATH_LIST="./object_detection_exp/ycbv/retinanet/retinanet_R_50_FPN_3x_edan_objects_sim_dropout01_lr5e-5/"
NUM_ITER=10
NUM_SEL=20
AGG_MODE="avg" # "max", 'sum', 'avg'
ACQ_MODE=("both" "cls_batch_bald" "cls") # 'cls', 'reg', 'both', 'max', 'cls_bald', 'cls_batch_bald' 
RND_LIST=("1" "2" "3") # ("1" "2" "3")
LR_LIST=( "1e-4") # "1e-3", "1e-4", "1e-5",
ES_LIST=("2000") # "2000", "1000"
ME_LIST=("0")  # max_epoch_list
SVAE_NAME="acq20_lr1e-4_bbald_clue_coreset_both_3rnds"
SAM_MODE=("RandomTopN" "clue" "Corset") # 'RandomTopN_clue', 'TopN', 'Coreset', 'RandomTopN', 'TopNBalancedSyn', 'TopNBalancedReal', 'RandomTopNBalancedSyn', 'RandomTopNBalancedReal'
LEG_SIZE="13"     
SUFFIX_LIST=("_WCls1.0_WReg0.01" ) # "_det250_WCls1.0_WReg0.01",  "_det15_WCls1.0_WReg0.01", //"_det1000", "_det200", // "", "_only_epistemic", "_only_aleatoric", "_both",
echo -e "1: "${ACQ_MODE[*]}"\n"
echo -e "SAM_MODE: "${SAM_MODE[*]}"\n"
echo -e "RND_LIST: "${RND_LIST[*]}"\n"
cmd="python $RUN_CODE --exp_folder_suffix_list ${SUFFIX_LIST[*]} --legend_size $LEG_SIZE --sampling_mode ${SAM_MODE[*]} --save_fig_name $SVAE_NAME --max_epoch_list ${ME_LIST} --early_stopping_list ${ES_LIST[*]} --lr_list ${LR_LIST[*]} --base_path_list ${PATH_LIST[*]} --num_iter $NUM_ITER --num_select $NUM_SEL --aggregate_mode $AGG_MODE --acquisition_mode ${ACQ_MODE[*]} --rnd_list ${RND_LIST[*]}"
    
echo -e "Executing: "$cmd"\n"
$cmd
