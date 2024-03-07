import pandas as pd
import datetime
class AciExampleProvider:
    def __init__(self, data_file, sample_size=15):
        self.data_file = data_file
        self.sample_size = sample_size
        self.meta_and_note = ""

    def sample_meta_and_note(self):
        """
        Samples metadata and note content from Aci.
        """
        try:
            aci_df = pd.read_csv(self.data_file)
            aci_df.rename(columns={'cc': 'chief_complaint'}, inplace=True)
            aci_data = aci_df[aci_df['dataset'] == 'aci']
            aci_data = aci_data.sample(n=self.sample_size)

            example_number = 1
            for index, row in aci_data.iterrows():
                metadata = row[-7:]
                note = row['note']

                self.meta_and_note += f"Example {example_number}:\n\nMetadata:\n"
                for col, value in metadata.items():
                    self.meta_and_note += f" {col}: {value}, "
                self.meta_and_note = self.meta_and_note.rstrip(', ')
                self.meta_and_note += f"\n\n\n\nNote:\n{note}\n\n{'*'*75}\n\n"
                example_number += 1

        except Exception as e:
            return f"An error occurred while sampling meta and note: {e}"
        
        return self.meta_and_note

    def write_to_file(self, output_file_base):
        """
        Just incase we wanna save the sampled data. 
        Writes the sampled meta and note to a file with a date-stamped filename.
        """
        current_date = datetime.datetime.now().strftime("%d%m")
        output_file = f"{output_file_base}_{current_date}.txt"
        try:
            with open(output_file, 'w') as file:
                file.write(self.meta_and_note)
            return f"Data has been written to {output_file}"
        except Exception as e:
            return f"An error occurred: {e}"