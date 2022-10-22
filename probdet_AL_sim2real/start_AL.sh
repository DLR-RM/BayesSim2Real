#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; ) # doesn't work on slurm cluster
cd $ROOT
echo '###! Current directory is:'
pwd
RUN_CODE="./sim2real_AL.py"

# need more attention to the following hyper-params
CONFIG_ROOT=${ROOT}"/configs"
AL_CONFIG="active_learning/YCBV_al_acq5_iter10_lr1e-3_uni_test.yaml" 
START_FROM=0
INF_CONFIG="Inference/bayes_od_mc_dropout.yaml" # bayes_od_mc_dropout, bayes_od, standard_nms, bayes_od_mc_dropout_30samples
AL_FOLDER_SUFFIX="_uni_test" # default should be "None"! "_val", "_coreset_sum_euc_unc0.01_val", "_RandomTopN_val", 
NOT_SAVE_PRED=0 # flag not to save predictions used during active learning 

# RNDSEED=1
echo -e "\nFirst argument to set RANDOM selection or not, 1 for random."
echo -e "Second and third arg to set the range of random seeds [\$2, \$3].\n"
RND=$1
if [ -z $1 ]
then
    echo "no input for RANDOM!"
    exit
fi 
RND=$1
echo "RANDOM is set to $RND,"
INITAL_RNDSEED=$2
FINAL_RNDSEED=$3
echo -e "RNDSEED is set to [$INITAL_RNDSEED, $FINAL_RNDSEED]!\n"

for RNDSEED in `seq $INITAL_RNDSEED $FINAL_RNDSEED`
do
    echo "Start training with Random seed "$RNDSEED
    if [ $RND -eq 0 ]
    then
        echo -e "RANDOM is FALSE\n"
        cmd="python $RUN_CODE --num-gpus 1 --not_save_pred $NOT_SAVE_PRED --inference-config $INF_CONFIG --al-config $AL_CONFIG --start_from $START_FROM --al_folder_suffix $AL_FOLDER_SUFFIX --random-seed $RNDSEED --config-file-root $CONFIG_ROOT"
        # python 
    else
        echo -e "RANDOM is TRUE\n"
        cmd="python $RUN_CODE --random --num-gpus 1 --not_save_pred $NOT_SAVE_PRED  --inference-config $INF_CONFIG --al-config $AL_CONFIG --start_from $START_FROM --al_folder_suffix $AL_FOLDER_SUFFIX --random-seed $RNDSEED --config-file-root $CONFIG_ROOT"
        # python  
    fi        
    
    echo -e "Executing: "$cmd"\n"
    $cmd
done
