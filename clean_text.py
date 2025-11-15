import json
import re

def clean_text(text, abbreviations_path):
    """
    Cleans a given text by:
    1. Converting to lowercase.
    2. Replacing common Vietnamese abbreviations using a provided JSON file.
    3. Removing emojis.
    4. Removing special characters (keeping only letters and spaces).
    5. Removing extra spaces.
    """
    if not isinstance(text, str):
        return ""

    try:
        with open(abbreviations_path, 'r', encoding='utf-8') as f:
            abbreviations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Abbreviations file not found at {abbreviations_path}")
        abbreviations = {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {abbreviations_path}")
        abbreviations = {}

    
    text = text.lower()


    sorted_abbr_keys = sorted(abbreviations.keys(), key=len, reverse=True)
    for abbr in sorted_abbr_keys:
        full_form = abbreviations[abbr]
        text = re.sub(r'\b' + re.escape(abbr.lower()) + r'\b', full_form.lower(), text)

    emoji_pattern = re.compile(
        "["  
        "\U0001F600-\U0001F64F"  
        "\U0001F300-\U0001F5FF"  
        "\U0001F680-\U0001F6FF"  
        "\U0001F1E0-\U0001F1FF"  
        "\U00002702-\U000027B0"  
        "\U000024C2-\U0001F251"  
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^0-9a-zàáạảãăằắặẳẵâầấậẩẫèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỳỹđ\s]', '', text, flags=re.UNICODE)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def clean_json_file(input_path, output_path, abbreviations_path):
    cleaned_data_final = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            
            if 'input' in item:
                item['input'] = clean_text(item['input'], abbreviations_path)

            if 'output' in item and isinstance(item['output'], dict) and 'infor' in item['output']:
                item['output']['infor'] = clean_text(item['output']['infor'], abbreviations_path)

            cleaned_data_final.append(item)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data_final, f, ensure_ascii=False, indent=4)
        print(f"Final cleaned data saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

