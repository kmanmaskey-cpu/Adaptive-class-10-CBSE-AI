import json
import os

# Define the file paths in one place so they are easy to change later
DATA_FILE = "questions.json"
MASTERY_FILE = "mastery.json"

def load_json_file(file_path):
    """A helper function to safely load any JSON file."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} does not exist.")
        return {} # Return an empty dictionary if file doesn't exist
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Warning: {file_path} is corrupted.")
        return {}
    
def load_mastery():
    return load_json_file(MASTERY_FILE)

def save_json_file(file_path, data):
    """A helper function to safely save data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"❌ Error saving to {file_path}: {e}")
