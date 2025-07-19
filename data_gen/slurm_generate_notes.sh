#!/bin/bash
#SBATCH --your_job_name
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-user=youremail@gmail.com  # Email address for notifications
#SBATCH --mail-type=ALL

#SBATCH --qos=normal

#SBATCH --time=3-00:00:00

#SBATCH -c 30
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "$(date): Job $SLURM_JOB_ID is allocated resource"

# Activate the conda environment

source /pkgs/anaconda3/bin/activate path_to_your_env || conda activate path_to_your_env


# put your command here
python generate_notes.py \
  --output_dir "path" \
  --num_icd10 50 \
  --notes_per_icd10 5



echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"
