#!/bin/bash
#SBATCH --job-name=curr_nli_electra_prototex
#SBATCH --output=slurm_execution/%x-%j.out
#SBATCH --error=slurm_execution/%x-%j.out
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/cluster/raid/home/himanshu.rawlani/propaganda_detection/prototex_custom
# Verify working directory
echo $(pwd)
# Print gpu configuration for this job
nvidia-smi
# Verify gpu allocation (should be 1 GPU)
echo $CUDA_VISIBLE_DEVICES
# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"
# Activate (local) env
conda activate prototex

echo "Starting training with parameters:"

# dataset="data/logical_fallacy_with_none"
# echo "dataset: ${dataset}"
# modelname="curr_finegrained_nli_electra_prototex"
# echo "modelname: ${modelname}"
# model_checkpoint="Models/curr_finegrained_nli_electra_prototex"
# 
# for trial in 3
# do
# echo "Trail number: ${trial}"
# python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "electra-prototex" --experiment "electra_finegrained_classification_$((trial))" --none_class "Yes" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra" --model_checkpoint ${model_checkpoint}
# done

dataset="../data/bigbench"
echo "dataset: ${dataset}"
modelname="curr_binary_nli_electra_prototex"
echo "modelname: ${modelname}"
model_checkpoint="../models/curr_binary_nli_electra_prototex"

for trial in 1
do
echo "Trail number: ${trial}"
python main.py --num_prototypes 50 --num_pos_prototypes 49 --data_dir ${dataset} --modelname ${modelname} --project "binary-electra-prototex" --experiment "electra_binary_classification_$((trial))" --none_class "No" --augmentation "Yes" --nli_intialization "Yes" --curriculum "Yes" --architecture "Electra" #--model_checkpoint ${model_checkpoint}
done

conda deactivate
