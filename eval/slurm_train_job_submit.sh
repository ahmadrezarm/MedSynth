#!/bin/bash
#SBATCH --job-name=D2N_remained_ablation_tuning
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-user=ahmad.rm0067@gmail.com  # Email address for notifications
#SBATCH --mail-type=ALL
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=1-00:00:00
#SBATCH -c 30
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "$(date): Job $SLURM_JOB_ID is allocated resource"

# Activate the conda environment

source /pkgs/anaconda3/bin/activate /h/ahmad/.conda/envs/SynthDataGenEnv || conda activate /h/ahmad/.conda/envs/SynthDataGenEnv


# put your command here
python /h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/eval/continous_eval/run_tuning.py

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"
