#!/bin/bash 
#SBATCH --job-name=phd_HPC_PyContinualExecution
##SBATCH --exclusive=user
##SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
##SBATCH --mem=30G
#SBATCH --time=unlimited
#SBATCH --mail-user=dionis@uo.edu.cu
#SBATCH --mail-type=END
#SBATCH --output=phd_HPC_PyContinualExecution_%A_%a.out # STDOUT
#SBATCH --error=phd_HPC_PyContinualExecution_%A_%a.err  # STDERR


case $SLURM_ARRAY_TASK_ID in
0) ARGS="--ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_experiment_phd --use_predefine_args" ;;
1) ARGS="--ntasks 8 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_wexperiment_phd --use_predefine_args" ;;
2) ARGS="--ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_fareway_restaurant_experiment_phd --use_predefine_args" ;;
3) ARGS="--ntasks 7 --bert_model bert-base-uncased --backbone bert_adapter --baseline classic --task asc --eval_batch_size 128 --train_batch_size 32 --scenario dil_classification --idrandom 0 --dataloaders asc_fareway_hotels_experiment_phd --use_predefine_args"  ;;
esac

module load Python/3.7.0-foss-2018b

cd  $SLURM_SUBMIT_DIR

conda activate phdContinualLearning

python3 -u run.py  $ARGS > phd_HPC_PyContinualExecution-$SLURM_ARRAY_TASK_ID.txt

#python3 run.py --bert_model 'bert-base-uncased' --backbone bert_adapter --baseline ctr --task asc --eval_batch_size 128 --train_batch_size 32 --scenario til_classification --idrandom 0  --use_predefine_args

