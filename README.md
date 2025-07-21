# MedSynth

## Environment Setup:
You can set up the Conda environment using the `environment.yml` file.

## To Generate Synthetic Medical Notes and Dialogues:
In our pipeline, notes need to be generated first, followed by the generation of dialogues based on those notes.
First, set your `OPENAI_API_KEY` as an environment variable.

### To Generate Synthetic Medical Notes:
1. Go to `data_gen/generate_notes.py` update the paths, and run the script. This will generate 5 notes per ICD-10 description, each saved as a separate `.csv` file in the specified directory.
If desired, you can change the number of notes per ICD-10 code by modifying `notes_count= 5`. 

### To Generate Corresponding Synthetic Dialogues:
1. Go to `code/data_gen/generate_dialogues.py` update the paths, and run the script. This will generate corresponding dialogues for the notes in the directory.
2. You can combine the resulting `.csv` files into a single `.csv` using `data_gen/extract_data_pairs_from_dfs.py`.



## To evaluate the generated dialogue-note pairs with traditionl metrics:
### For Dialogue-2-Note task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `eval/continous_eval/dial2_note_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `eval/continous_eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `eval/continous_eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `eval/continous_eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `eval/continous_eval/slurm_eval_job_submit.sh`.
4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `eval/continous_eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.

### For Note-2-Dialogue task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `eval/continous_eval/note2dial_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `eval/continous_eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `eval/continous_eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `eval/continous_eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `eval/continous_eval/slurm_eval_job_note2dial.sh`.
4. 4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `eval/continous_eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.


## To evaluate the generated dialogue-note pairs with the Jury:
### GPT-4o as the judge:
1. Go to `eval/continous_eval/get_promethus_relative_score_GPT.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.

### Prometheus as the judge:
1. Go to `eval/continous_eval/get_promethus_relative_score.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.

### Qwen as the judge:
1. Go to `eval/continous_eval/get_qwen_relative_scores.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.


