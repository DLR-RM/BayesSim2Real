#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RNDSEED=1
CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_ycbv_sim_dropout01_lr5e-4_unitest.yaml"
# CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_ycbv_real_dropout01_lr5e-5_unitest.yaml"
# CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_sam_sim_dropout01_lr5e-5.yaml"
# CONFIG="configs/retinanet/retinanet_R_50_FPN_3x_edan_objects_real_dropout01_nlldiag_lr5e-5.yaml"

cmd="python train_net.py --num-gpus 1 --random-seed $RNDSEED --config-file $CONFIG --resume" 
echo -e "Executing: "$cmd"\n"
$cmd
