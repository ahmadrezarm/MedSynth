from utils import model_tuner
from utils import constants

tuning_model_dict= {
                   "N2D_ACI_Ahmad_246_Aci": "Ahmad0067/SynthDataGen_llama3_N2D_ACI_Ahmad_246_Samples_and_Aci_instruct_2025-02-08_11-32",
                   "N2D_ACI_AllQwen_and_Aci": "Ahmad0067/SynthDataGen_llama3_N2D_ACI_AllQwen_and_Aci_instruct_2025-02-08_11-26",
                   "N2D_ACI_AllQwen_57_Samples_and_Aci": "Ahmad0067/SynthDataGen_llama3_N2D_ACI_AllQwen_57_Samples_and_Aci_instruct_2025-02-08_11-25",
                   }


def main():
    for model, hf_path in tuning_model_dict.items():
        print(f"working on: {model}")
        print(f" The datset is: {hf_path}")
        print(20*"*")
        tuner = model_tuner.ModelTuner(TRAINING_DATA_PATH_HF= hf_path, 
                                  FINE_TUNED_MODEL_NAME= f"SynthDataGen_{model}", 
                                  tuning_config= constants.tuning_config, base_model= constants.base_model)
    
        tuner.model_train_and_save()
        print(f"Model: SynthDataGen_{model}  has been pushed to the hub successfully. Going to the next model ...")
        print(20*"#")


if __name__ == '__main__':
    main()

