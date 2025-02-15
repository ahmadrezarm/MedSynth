from utils import notechat_sampler



def main():
    num_samples= 246
    sampled_df= notechat_sampler.get_sample_note_chat(num_samples= num_samples)
    notechat_sampler.save_note_chat_sample_to_csv(sampled_df, num_samples= num_samples)


if __name__ == '__main__':
    main()
