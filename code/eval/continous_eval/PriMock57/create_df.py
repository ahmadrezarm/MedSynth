import os
import json
import pandas as pd
from pathlib import Path

# Define the paths to the notes and dialogues folders
notes_folder = Path('/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/notes_correct')        # e.g., Path('/home/user/notes_folder')
dialogues_folder = Path('/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/transcripts_processed')  # e.g., Path('/home/user/dialogues_folder')

# Initialize lists to store the extracted notes and dialogues
notes_list = []
dialogues_list = []

# Iterate through each JSON file in the notes folder
for note_file in notes_folder.glob('*.json'):
    # Extract the base filename without extension
    base_name = note_file.stem
    
    # Construct the corresponding dialogue file path
    dialogue_file = dialogues_folder / f"{base_name}.txt"
    
    # Check if the corresponding dialogue file exists
    if dialogue_file.exists():
        try:
            # Open and read the JSON file
            with open(note_file, 'r', encoding='utf-8') as nf:
                data = json.load(nf)
                # Extract the 'note' value
                note = data.get('note', '').strip()
            
            # Open and read the dialogue text file
            with open(dialogue_file, 'r', encoding='utf-8') as df:
                dialogue = df.read().strip()
            
            # Append the extracted data to the lists
            notes_list.append(note)
            dialogues_list.append(dialogue)
            
            print(f"Processed pair: {base_name}")
        
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {note_file}")
        except Exception as e:
            print(f"An error occurred while processing {base_name}: {e}")
    else:
        print(f"Dialogue file not found for: {base_name}")

# Create a pandas DataFrame with the extracted notes and dialogues
df = pd.DataFrame({
    'notes': notes_list,
    'dialogues': dialogues_list
})

# Define the output CSV file path
output_csv = Path('/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/unified_data/primock57.csv')  # e.g., Path('/home/user/output.csv')

# Ensure the output directory exists
output_csv.parent.mkdir(parents=True, exist_ok=True)

# Save the DataFrame to CSV
df.to_csv(output_csv, index=False, sep='|')  # Using '|' as the separator
print(f"Data successfully saved to {output_csv}")
