from utils import model_tuner
from utils import constants


# change this whenever a new model is being trained on a new data
def main():
    tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/SynthDataGen_llama3_Note2Dial_ACI_Ahmad56_Only_instruct_dataset2024-12-09_08-19", 
                                  FINE_TUNED_MODEL_NAME= "SynthDataGen_Note2Dial_llama-3-8b-Instruct_Ahmad_56_Only", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
    tuner.model_train_and_save()


if __name__ == '__main__':
    main()

