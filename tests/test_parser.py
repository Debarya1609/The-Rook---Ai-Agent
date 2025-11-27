import json
import os
import pytest
import glob

# Assume you have a function to test, e.g., 'parse_json_data'
def parse_json_data(filepath):
    """A placeholder function that your test will verify."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Add your specific parsing logic here
    return data

def json_files():
    """Fixture to collect all JSON file paths from the logs directory."""
    # Adjust the path to your test data directory
    #data_dir = os.path.join(os.path.dirname(__file__), '..', 'logs/llm_samples')
    # Use glob to find all files ending with .json recursively
    #files = glob.glob(os.path.join(data_dir, '*.json'), recursive=True)
    files = glob.glob('logs/**/*.json', recursive=True)
    return files

@pytest.mark.parametrize("file_path", json_files())
def test_extract_json(file_path):
    """Test the parser function for each JSON file."""
    print(f"\nTesting with file: {os.path.basename(file_path)}")
    try:
        data = parse_json_data(file_path)
        # Add your assertions here to validate the extracted data
        assert isinstance(data, dict) or isinstance(data, list), "Parsed data should be a dict or list"
        assert len(data) > 0, "Parsed data should not be empty"
        # Example assertion: check for a specific key if your JSON structure is consistent
        # assert "expected_key" in data, "Missing expected_key in JSON data"
    except json.JSONDecodeError:
        pytest.fail(f"Failed to decode JSON from file: {file_path}")
    except Exception as e:
        pytest.fail(f"An error occurred while processing file {file_path}: {e}")
