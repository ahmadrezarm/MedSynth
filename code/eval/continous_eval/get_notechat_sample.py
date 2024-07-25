from utils import notechat_sampler



def main():
    sampled_df= notechat_sampler.get_sample_note_chat(num_samples= 1500)
    notechat_sampler.save_note_chat_sample_to_csv(sampled_df)


if __name__ == '__main__':
    main()
