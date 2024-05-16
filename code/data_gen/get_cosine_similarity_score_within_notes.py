from gen_utils import semantic_similarity_functions


def main():
    df_with_embeding= semantic_similarity_functions.embed_scenarios_and_save("/h/ahmad/SynthDataGen/Synthetic_Data_Gen/data/output/notes_onVector/CHRONIC PAIN SYNDROME_2024-05-15")
    average_score_rejected_excluded= semantic_similarity_functions.get_cosine_similarity(df= df_with_embeding,
                                                                                            mode= "exclude_rejected")
    
    average_score_rejected_included= semantic_similarity_functions.get_cosine_similarity(df= df_with_embeding,
                                                                                            mode= "include_rejected")






if __name__ == '__main__':
    main()
