#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RNDSEED=1
NUM_ITER=10
RUN_CODE="sim2real_AL_only_eval.py"
CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_ycbv_sim_dropout01_lr5e-4_unitest.yaml"
INFER_CONFIG="configs/Inference/standard_nms.yaml"
# INFER_CONFIG="configs/Inference/bayes_od_cls_mc_dropout.yaml"
TEST_SET="ycbv_real_test" # "ycbv_real_test_all", "ycbv_real_test"
AL_EXP_NAME="acq5_iter10_cls_batch_bald_1000_rnd1_eval5_lr0.001_uni_test" # "acq50_iter10_rnd1_eval2_lr5e-4_RANDOM"

cmd="python $RUN_CODE --num-gpus 1 --al_exp_name $AL_EXP_NAME --random-seed $RNDSEED --num_iter $NUM_ITER --config-file $CONFIG --test-dataset $TEST_SET --inference-config $INFER_CONFIG" 
echo -e "Executing: "$cmd"\n"
$cmd
