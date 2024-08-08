from utils import model_tuner
from utils import constants


# change this whenever a new model is being trained on a new data
def main():
    tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/SynthDataGen_llama3_Ahmad_4000_Sample_and_Aci_train_instruct_dataset2024-07-31_20-18", 
                                  FINE_TUNED_MODEL_NAME= "SynthDataGen_Dial2Note_llama-3-8b-Instruct_Ahmad_4000_samples_and_Aci", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
    tuner.model_train_and_save()


if __name__ == '__main__':
    main()

