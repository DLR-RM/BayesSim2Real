#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RUN_CODE="draw_results_json.py"
EXP_FOLDER=$ROOT"/classification_exp/al_training/"
OUT_PATH=$ROOT"/classification_exp/results_figs/acq20_lr1e-5_1e-3_warmup_refined_v2.pdf"
cmd="python $RUN_CODE --res-path $EXP_FOLDER --out-path $OUT_PATH"
echo -e "Executing: "$cmd"\n"
$cmd
