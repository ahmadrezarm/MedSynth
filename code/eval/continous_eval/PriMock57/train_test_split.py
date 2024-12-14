import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    input_csv_path = Path('/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/unified_data/primock57.csv')  # Update this path as needed

    # Define the output directory for split CSV files
    output_dir = Path('/h/ahmad/SynthDataGen/Synthetic_Data_Gen/code/eval/continous_eval/PriMock57/unified_data/train_test_split')
    output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the paths for the train and test CSV files
    train_csv_path = output_dir / 'train.csv'
    test_csv_path = output_dir / 'test.csv'

    # Define the test size and random state for reproducibility
    test_size = 0.35  
    random_state = 42  
    # Load the combined CSV file into a pandas DataFrame
    try:
        df = pd.read_csv(input_csv_path, sep='|')
        print(f"Successfully loaded data from {input_csv_path}")
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} does not exist.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_csv_path} is empty.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_csv_path}: {e}")
        exit(1)

    # Perform the train-test split
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True  # Shuffle the data before splitting
        )
        print(f"Train-Test split completed with test size = {test_size*100}%")
        print(f"Training samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")
    except Exception as e:
        print(f"An error occurred during train-test split: {e}")
        exit(1)

    # Save the training DataFrame to a CSV file
    try:
        train_df.to_csv(train_csv_path, index=False, sep='|')
        print(f"Training data saved to {train_csv_path}")
    except Exception as e:
        print(f"An error occurred while saving training data to {train_csv_path}: {e}")
        exit(1)

    # Save the testing DataFrame to a CSV file
    try:
        test_df.to_csv(test_csv_path, index=False, sep='|')
        print(f"Testing data saved to {test_csv_path}")
    except Exception as e:
        print(f"An error occurred while saving testing data to {test_csv_path}: {e}")
        exit(1)

if __name__ == '__main__':
    main()
