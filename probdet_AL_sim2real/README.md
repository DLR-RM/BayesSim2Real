# probdet_syn2real
project to investigate active learning with Bayesian object detector for Sim-to-Real transfer.

### Code Structure
```
├── configs
│   ├── Base-COCO-RCNN-FPN.yaml
│   ├── Base-Inference.yaml
│   ├── active_learning # config files for active learning for sim2real
│   ├── faster-rcnn # config files for initial training of faster-rcnn
│   ├── Inference
│   └── retinanet # config files for initial training of retinanet (this work)
├── src
│   ├── __init__.py
│   ├── active_learning_src     # active learning 
│   ├── core                    # scripts for data sets and configs
│   ├── probabilistic_inference # inference with Bayes OD
│   └── probabilistic_modeling  # training with dropouts
├── apply_net.py                # teval a single model
├── read_imbalance_ratio.py     # compute imbalance ratio over iterations in AL
├── read_json_al_results.py     # functions to read results
├── sim2real_AL_only_eval.py    # eval active learning
├── sim2real_AL.py              # start active learning
├── train_net.py                # initial training 
├── README.md
├── draw_al_results.sh
├── eval_al_net.sh
├── start_AL.sh
├── test_net.sh
└── train_net.sh
```

Dataset paths are set in `src/core/datasets/setup_datasets.py` and `src/active_learning_src/active_learner.py/Active_learner.__init_pool_set()`.

### How to start

#### Initial training on Simulation Dataset

`train_net.sh`: to start normal training;

`test_net.sh`: to start normal evaulation;

#### Start Active Learning on Real Dataset

`start_AL.sh`: to start active learning;

#### Evaluation

`eval_al_net.sh`: to evaluate active learning;

`draw_al_results.sh`: to draw plots from the experiment results folder;

#### Experiments Folder Structure
The folder of experiments will be generated in the following format by executing the aforementioned commands:
```
retinanet_R_50_FPN_3x_edan_objects_sim_dropout/ 
├── AL_acq20_iter10_avg_both_rnd1                                                                
│   ├── al_results.json      # results of mAP in each iteration                   
│   ├── config.yaml                             
│   ├── iter1                # training log and saved model of each iteration               
│   ├── iter2   
|   |   iter...
│   ├── log.txt              # logs of the whole loop                
│   └── pretrained_inference # evaluation before iter1
├── last_checkpoint                          
├── log.txt                                  
├── metrics.json                        
├── model_final.pth          # the model pre-trained on synthetic data
```
