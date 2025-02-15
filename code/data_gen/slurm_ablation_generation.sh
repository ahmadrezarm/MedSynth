#!/bin/bash
#SBATCH --job-name=Ab_all_qwen_dial
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-user=ahmad.rm0067@gmail.com  # Email address for notifications
#SBATCH --mail-type=ALL

##SBATCH --qos=normal

#SBATCH --account=deadline
#SBATCH --qos=deadline

#SBATCH --time=3-00:00:00

#SBATCH -c 30
#SBATCH --mem=150G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "$(date): Job $SLURM_JOB_ID is allocated resource"

# Activate the conda environment

source /pkgs/anaconda3/bin/activate /h/ahmad/.conda/envs/SynthDataGenEnv || conda activate /h/ahmad/.conda/envs/SynthDataGenEnv


# put your command here
python /h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/data_gen/generate_dialogues.py
#/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/data_gen/generate_notes.py



echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"
