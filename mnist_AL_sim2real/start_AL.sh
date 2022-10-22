#!/bin/sh
ROOT=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $ROOT
echo '###! Current directory is:'
pwd
RUN_CODE="run_exp.py"

# need more attention to the following hyper-params
EXP_NAME="acq20_lr1e-5_warmup_ep35_ent_clue"
PROB_SCORE=0
SS_RATIO=100  # percentage to be kept after subsampling before scoring
EP=35 # max epochs to train
LR=1e-5
EARLY_STOP=35
NUM_MC_SAMPLES=20   
ACQ_SIZE=20
TOTAL_ACQ_SIZE=1000
ACQ_FUNC="predictive_entropy" # "random", "bald", "predictive_entropy",  
ACQ_METHOD="clue" #  "independent", "multibald", "clue"

# Customized folders
EXP_ROOT="./classification_exp/al_training_logs"
MODEL_PTH="./classification_exp/pretrained_models/mnist"

# RNDSEED=1
echo -e "First and second arg to set the range of random seeds [\$2, \$3] e.g. [1, 3].\n"
INITAL_RNDSEED=$1
FINAL_RNDSEED=$2
echo -e "RNDSEED is set to [$INITAL_RNDSEED, $FINAL_RNDSEED]!\n"

for RNDSEED in `seq $INITAL_RNDSEED $FINAL_RNDSEED`
do
    echo "Start training with Random seed "$RNDSEED
    if [ $PROB_SCORE -eq 0]
    then
        cmd="python $RUN_CODE --exp-root $EXP_ROOT --model-path $MODEL_PTH --balanced_validation_set --seed $RNDSEED --type $ACQ_FUNC --acquisition_method $ACQ_METHOD --epochs $EP --dataset mnistm --initial_sample -1 --initial_percentage $SS_RATIO --experiment_task_id $EXP_NAME --learning_rate $LR --early_stopping_patience $EARLY_STOP --num_inference_samples $NUM_MC_SAMPLES --available_sample_k $ACQ_SIZE --target_num_acquired_samples $TOTAL_ACQ_SIZE --target_accuracy 0.95"
    else
        cmd="python $RUN_CODE --exp-root $EXP_ROOT --model-path $MODEL_PTH --prob_score_sampling --balanced_validation_set --seed $RNDSEED --type $ACQ_FUNC --acquisition_method $ACQ_METHOD --epochs $EP --dataset mnistm --initial_sample -1 --initial_percentage $SS_RATIO --experiment_task_id $EXP_NAME --learning_rate $LR --early_stopping_patience $EARLY_STOP --num_inference_samples $NUM_MC_SAMPLES --available_sample_k $ACQ_SIZE --target_num_acquired_samples $TOTAL_ACQ_SIZE --target_accuracy 0.95"
    fi

    echo -e "Executing: "$cmd"\n"
    $cmd
done
