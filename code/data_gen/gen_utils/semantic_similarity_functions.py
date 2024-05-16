# useful links: 
# 1. https://www.datacamp.com/tutorial/introduction-to-text-embeddings-with-the-open-ai-api
# 2. https://cookbook.openai.com/examples/get_embeddings_from_dataset
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity


from .note_generation_functions import initialize_openai_client
from . import gen_constants

client= initialize_openai_client()
def get_embedding(text, client,
                  model=gen_constants.embeding_model):
   
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding


def embed_scenarios_and_save(path_to_df):
    df = pd.read_csv(path_to_df, sep="|")
    df['embedding'] = df.Scenario.apply(lambda x: get_embedding(text=x, client= client))
    df['embedding'] = df['embedding'].apply(json.dumps)
    df.to_csv(f"{path_to_df}_with_embeding.csv", sep="|", index= False)
    print("Embeding generated and saved successfully!")
    return df


def get_cosine_similarity(df, mode):
    if mode == "exclude_rejected":
        # Filter out rejected notes
        df_filtered = df[df["Note"] != "Rejected"]
    elif mode == "include_rejected":
        # Include all notes
        df_filtered = df
    else:
        raise ValueError("Invalid mode. Choose either 'only_approved' or 'include_rejected'.")

    # Convert JSON strings back to lists
    df_filtered['embedding'] = df_filtered['embedding'].apply(json.loads)
    df_filtered['embedding'] = df_filtered['embedding'].apply(lambda x: np.array(x, dtype=float))
    # Stack embeddings into a 2D array
    embeddings = np.vstack(df_filtered['embedding'].values)

    similarity_matrix = cosine_similarity(embeddings)

    # Exclude self-similarities (diagonal elements)
    np.fill_diagonal(similarity_matrix, np.nan)

    # Calculate the average score, excluding NaN values
    average_score = np.nanmean(similarity_matrix)
    print(f"Average similarity score for mode: {mode} is: {average_score}")

    return average_score


