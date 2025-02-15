# Synthetic_Data_Gen

# Synthetic_Data_Gen

## Envriornment setup:
You can set up the conda envriorment using `environment.yml` file.

## To generate synthetic medical notes and dialogues:
In our pipelines, notes need to be generated first and then the dialogues can be generated based on the notes. 
First, Set your OPENAI_API_KEY as an env variable

### To generate synthetic medical notes:
1. Go to `code/data_gen/generate_notes.py` and update the paths, then run it. You will have 5 notes per ICD-10 description, each saved into a separate .csv file in  the directory. If you want, you can change the number of notes per ICD-10 by changing `notes_count= 5`. 

### To generate corresponding synthetic dialogoues:
1. Go to `code/data_gen/generate_dialogues.py` and update the paths, then run it. This will generate the corresponding dialogues for the notes in the directory. 
2. You can combine the resulting .csv files into a single .csv using `code/data_gen/extract_data_pairs_from_dfs.py`.



## To evaluate the generated dialogue-note pairs with traditionl metrics:
### For Dialogue-2-Note task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `code/eval/continous_eval/dial2_note_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `code/eval/continous_eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `code/eval/continous_eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `code/eval/continous_eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `code/eval/continous_eval/slurm_eval_job_submit.sh`.
4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `code/eval/continous_eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.

### For Note-2-Dialogue task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `code/eval/continous_eval/note2dial_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `code/eval/continous_eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `code/eval/continous_eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `code/eval/continous_eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `code/eval/continous_eval/slurm_eval_job_note2dial.sh`.
4. 4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `code/eval/continous_eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.


## To evaluate the generated dialogue-note pairs with the Jury:
### For Dialogue-2-Note task:
1. 
