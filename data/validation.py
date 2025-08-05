import json
import pandas as pd
from pathlib import Path

def validate_qa_pairs(validation_file, training_dir):
    """
    Validates question/answer pairs from training data against a validation set.

    Args:
        validation_file (str): The path to the validation JSON file.
        training_dir (str): The directory containing the training JSON files.
    """
    try:
        # Load the validation data
        with open(validation_file, 'r') as f:
            validation_data = json.load(f)
        df_validation = pd.DataFrame(validation_data)
        df_validation_unique = df_validation.drop_duplicates(subset=['question', 'answer'])

        # Aggregate training data
        all_training_data = []
        training_path = Path(training_dir)
        for file_path in training_path.glob('*_qa_pairs.json'):
            with open(file_path, 'r') as f:
                all_training_data.extend(json.load(f))

        if not all_training_data:
            print("No training files found or files are empty.")
            return

        df_training = pd.DataFrame(all_training_data)
        df_training_unique = df_training.drop_duplicates(subset=['question', 'answer'])

        # Find matching pairs
        df_matches = pd.merge(df_training, df_validation, on=['question', 'answer','image_file'], how='inner')

        # Calculate and print accuracy
        total_training_pairs = len(df_training_unique)
        matching_pairs = len(df_matches)
        total_valid_pairs = len(df_validation_unique)
        print("total records in validation :",total_valid_pairs)
        #print(" matching_pairs :",df_matches)
        print("total training records :",len(df_training))
        print("total matching records :",len(df_matches))
        print("total valid records :",len(df_validation))

        if total_training_pairs > 0:
            accuracy = (len(df_matches) / len(df_validation)) * 100
            print(f"Total unique training pairs: {total_training_pairs}")
            print(f"Matching pairs found in validation data: {matching_pairs}")
            print(f"Accuracy of the training data compared to the validation data: {accuracy:.2f}%")
        else:
            print("No unique question/answer pairs found in the training data to validate.")

        if not df_matches.empty:
            print("\nMatching question/answer pairs:")
            print(df_matches.head())
        else:
            print("\nNo matching question/answer pairs were found.")


        # Find records where image and question match, but answers differ
        df_mismatched = pd.merge(
            df_training,
            df_validation,
            on=['image_file', 'question'],
            suffixes=('_training', '_validation'),
            how='inner'
        )

        df_mismatched = df_mismatched[df_mismatched['answer_training'] != df_mismatched['answer_validation']]
        print(" mismatched", df_mismatched)
        if not df_mismatched.empty:
            print("Found records with matching image and question but different answers:")
            for _, row in df_mismatched.iterrows():
                print("-" * 20)
                print(f"Image File: {row['image_file']}")
                print(f"Question: {row['question']}")
                print(f"Training Answer: {row['answer_training']}")
                print(f"Validation Answer: {row['answer_validation']}")
        else:
            print("No records with mismatched answers were found.")

    except FileNotFoundError as e:
        print(f"Error: A file was not found. Please check the paths: {e}")
    except json.JSONDecodeError:
        print("Error: Could not decode one of the JSON files. Please check the file format.")


# Example usage (you will need to change the paths)
# validate_qa_pairs('balanced_qa_pairs.json', '/data/train')

validate_qa_pairs('balanced_qa_pairs.json','valid')