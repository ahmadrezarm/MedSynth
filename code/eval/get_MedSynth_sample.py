from utils import MedSynth_hf_sampler



def main():
    num_samples= 8500
    sampled_df= MedSynth_hf_sampler.get_sample_AhmadData(num_samples= num_samples)
    MedSynth_hf_sampler.save_AhmadData_sample_to_csv(sampled_df, num_samples= num_samples)


if __name__ == '__main__':
    main()
