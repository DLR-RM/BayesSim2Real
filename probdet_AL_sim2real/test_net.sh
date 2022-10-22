#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RNDSEED=1
RUN_CODE="apply_net.py"
# training config
CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_ycbv_real_dropout01_lr5e-5_unitest.yaml"

# Deterministic
# INFER_CONFIG="configs/Inference/standard_nms.yaml"
# MC dropout
INFER_CONFIG="configs/Inference/bayes_od_mc_dropout.yaml"
TEST_SET="ycbv_real_test" # "ycbv_real_test_all", "ycbv_real_test"

cmd="python $RUN_CODE --num-gpus 1 --random-seed $RNDSEED --config-file $CONFIG --test-dataset $TEST_SET --inference-config $INFER_CONFIG" 
echo -e "Executing: "$cmd"\n"
$cmd
