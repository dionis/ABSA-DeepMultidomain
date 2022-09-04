#!/bin/bash 

python3 -u run.py --ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_experiment_phd --use_predefine_args --save_model --num_train_epochs 10 --save_each_step --aux_net --experiment same_phd_taskplus>> outuput_asc_experiment_phd.txt &
python3 -u run.py --ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args --save_model --num_train_epochs 10 --save_each_step --aux_net --experiment same_phd >> output_asc_wexperiment_phd.txt &
python3 -u run.py --ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args --num_train_epochs 10 --common_prmtrs --save_model --save_each_step --aux_net --experiment same_parameters >> output_asc_same_parameters.txt &
python3 -u run.py --ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args --num_train_epochs 10  --common_prmtrs --save_model --save_each_step --aux_net --experiment invert_input_sameparameters >> output_asc_invert_input_sameparameters.txt &

###########################################################################################
#
#
## Importan use in paperspace (parameters) because GPU memory explode
#
# !python3 -u run.py --ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 32 --train_batch_size 16 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args --num_train_epochs 10  --common_prmtrs --save_model --save_each_step --aux_net --experiment invert_input_sameparameters
#
#
###########################################################################################
