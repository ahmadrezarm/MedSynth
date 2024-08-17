from bench_utils import constants, model_tuner



# change this whenever a new model is being trained on a new data
def main():
    tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/SynthDataGen_Benchmarking_mistral_instruct_dataset_train_2024-08-15_13-47", 
                                  FINE_TUNED_MODEL_NAME= "SynthDataGen_Benchmarking_mistral-7b-instruct-v0.3", 
                                  tuning_config= constants.tuning_config, base_model= "unsloth/mistral-7b-instruct-v0.3") # "unsloth/llama-3-8b-Instruct"
    
    tuner.model_train_and_save()


if __name__ == '__main__':
    main()

