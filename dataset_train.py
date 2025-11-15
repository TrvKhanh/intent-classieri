from clean_text import clean_json_file
from datasets import Dataset
import json

# Paths to input data, output cleaned data, and abbreviations mapping
PATH_DATA = "./data.json"                
OUTH_PATH = "./data_train.json"          
ABBREVIATIONS_PATH = "./abbreviations.json"  
output_cleaned_path = clean_json_file(PATH_DATA, OUTH_PATH, ABBREVIATIONS_PATH)

max_input_length = 128
max_target_length = 64

def load_dataset(path=output_cleaned_path):
    """
    Load the cleaned JSON data and convert it into a HuggingFace Dataset object
    suitable for seq2seq model training.

    Steps:
    1. Read cleaned JSON file from disk.
    2. Format each item as a dictionary with 'input_text' and 'target_text'.
       - 'input_text' is prepended with a task prefix 'classify:' to indicate
         the task to the model.
       - 'target_text' is converted to a JSON string to preserve structured output.
    3. Construct a Dataset from a dictionary of lists.
    
    Returns:
        datasets.Dataset: A HuggingFace Dataset object with 'input_text' and 'target_text'.
    """
    # Load the cleaned JSON file
    with open(output_cleaned_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    training_data_json = []
    for item in cleaned_data:
        # Prefix input with task description; ensures model knows this is a classification task
        input_text = f"classify: {item['input']}"
        # Convert output to JSON string to preserve structure and avoid tokenization issues
        target_text = json.dumps(item['output'], ensure_ascii=False)
        training_data_json.append({
            "input_text": input_text, 
            "target_text": target_text
        })

    # Convert list of dicts to dict of lists to feed into HuggingFace Dataset
    data_dict = {
        "input_text": [item["input_text"] for item in training_data_json],
        "target_text": [item["target_text"] for item in training_data_json],
    }

    # Return as a Dataset object compatible with transformers training pipeline
    return Dataset.from_dict(data_dict)
