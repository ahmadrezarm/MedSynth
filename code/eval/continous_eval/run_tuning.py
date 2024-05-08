from utils import tune_model, constants

def main():
    tuner = tune_model.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/llama3_TaskC-TrainingSet_SOAP_instruct_dataset", 
                                  FINE_TUNED_MODEL_NAME= "llama-3-8b-Instruct-aci-train", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
    tuner.model_train_and_save()


if __name__ == '__main__':
    main()