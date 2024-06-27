import pandas as pd
import os

directory = "/Users/ahmadrezaie/papers/Synthetic_Data_Gen/data/output/notes_onVector/Phase_2/500_sample/"

all_files = os.listdir(directory)

#filtered_files = [file for file in all_files if file.endswith("_with_dial.csv")]

df_list = []

# Loop through the filtered files and read the required columns
for file in all_files: #filtered_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, usecols=["Role", "Disease Description", "Polished Note", "Abbreviated Note"], sep="|") #"polished_dial"
    df_list.append(df)

# Concatenate all dataframes into one
combined_df = pd.concat(df_list, ignore_index=True)
combined_df= combined_df[combined_df["Polished Note"] != "Rejected"]


combined_df.to_csv(f"{directory}_only_notes_combined.csv", sep="|", index=False)