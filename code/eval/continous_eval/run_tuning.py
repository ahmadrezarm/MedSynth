from utils import model_tuner
from utils import constants


# change this whenever a new model is being trained on a new data
def main():
    tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= "Ahmad0067/SynthDataGen_llama3_NoteChat_Sample_instruct_dataset_v32024-06-06_19-45", 
                                  FINE_TUNED_MODEL_NAME= "llama-3-8b-Instruct-bnb-4bit-aci_note-chat_v3", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
    tuner.model_train_and_save()


if __name__ == '__main__':
    main()