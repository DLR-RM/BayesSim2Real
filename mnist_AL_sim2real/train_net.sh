#!/bin/sh
conda activate probdet
PYTHON="python"

ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RUN_CODE="run_exp_no_al.py"
EXP_ROOT="./classification_exp"
DATASET="mnistm" # mnistm, mnist

cmd="python $RUN_CODE --dataset $DATASET --exp-root $EXP_ROOT"
echo "executing $cmd"
$cmd
