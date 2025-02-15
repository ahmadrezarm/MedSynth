#!/bin/bash
#SBATCH --job-name=Prom_D2N_relative_remained
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-user=ahmad.rm0067@gmail.com  # Email address for notifications
#SBATCH --mail-type=ALL
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=1-00:00:00
#SBATCH -c 30
#SBATCH --mem=40G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "$(date): Job $SLURM_JOB_ID is allocated resource"

# Activate the conda environment

source /pkgs/anaconda3/bin/activate /h/ahmad/.conda/envs/SynthDataGenEnv || conda activate /h/ahmad/.conda/envs/SynthDataGenEnv

# put your command here #########   GPT
python /h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/eval/continous_eval/get_promethus_relative_score.py
#/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/eval/continous_eval/get_promethus_relative_score_GPT.py
#
#




echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"
