# MedSynth

## Environment Setup:
You can set up the Conda environment using the `environment.yml` file.

## To Generate Synthetic Medical Notes and Dialogues:
In our pipeline, notes need to be generated first, followed by the generation of dialogues based on those notes.
First, set your `OPENAI_API_KEY` as an environment variable.

### To Generate Synthetic Medical Notes:
You can use the code below to generte notes: 
```python generate_notes.py \
  --output_dir "path" \
  --icd_csv_path "path"\
  --aci_train_path "path" \
  --num_icd10 50 \
  --notes_per_icd10 5 ```

Alternatively, you can use `slurm_generate_notes.sh` to generate notes. These will generate 5 notes per ICD-10 description by default, each saved as a separate `.csv` file in the specified directory.
If desired, you can change the number of notes per ICD-10 code by modifying `notes_count= 5`. 

### To Generate Corresponding Synthetic Dialogues:
You can use the follwing code: 
```python generate_dialogues.py \
  --path_to_input "path" \
  --aci_train_path "path" ```

Alternatively, you can use `slurm_generate_dialogue.sh`. These will generate corresponding dialogues for the notes in the directory.
You can combine the resulting `.csv` files into a single `.csv` using `data_gen/extract_data_pairs_from_dfs.py`.



## To evaluate the generated dialogue-note pairs with traditionl metrics:
### For Dialogue-2-Note task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `eval/dial2_note_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `eval/slurm_eval_job_submit.sh`.
4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `eval/continous_eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.

### For Note-2-Dialogue task:
1. First, the generated data must be converted into the required format for instrcution fine-tuning and be saved to huggingface. To do so, go to `eval/note2dial_create_process_datasets.py` and update the paths, then run it. It will create the dataset ready for instruction fine-tuning of Llama 3 and upload it to huggingface.
2. Go to `eval/run_tuning.py` and update the paths, then run it. It will tune the model and save the tuned model to huggingface. Alternatively, you can edit and use `eval/slurm_train_job_submit.sh`  to submit the tuning job to a slurm cluster. 
3. Go to `eval/run_eval.py` and edit the paths, then run it. It will evaluate the tuned model and return the model responses, as well as automatic metrics. Alternatively, you can submit the job to a slurm cluster using `eval/slurm_eval_job_note2dial.sh`.
4. 4. To repeat the same process with NoteChat dataset, you need to first take a sample from it with the same size as your generated data using `eval/get_notechat_sample.py`. Then repeat steps 1 to 3 for tuning and evaluation.


## To evaluate the generated dialogue-note pairs with the Jury:
### GPT-4o as the judge:
1. Go to `eval/get_promethus_relative_score_GPT.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.

### Prometheus as the judge:
1. Go to `eval/get_promethus_relative_score.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.

### Qwen as the judge:
1. Go to `eval/get_qwen_relative_scores.py`, and edit the paths. Also, depending on the task (dial-2-note or note-2-dial), uncomment the correct main function and run it.


