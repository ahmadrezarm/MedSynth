#!/bin/bash

#SBATCH --qos=a100_ahmadrm
#SBATCH --partition=a100


echo "$(date): Job $SLURM_JOB_ID is allocated resource"

# Activate the conda environment

source /pkgs/anaconda3/bin/activate /h/ahmad/.conda/envs/SynthDataGenEnv || conda activate /h/ahmad/.conda/envs/SynthDataGenEnv



python /h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/data_gen/generate_notes.py
#/h/ahmad/SynthDataGen_v2/Synthetic_Data_Gen/code/data_gen/generate_dialogues.py
#



echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"