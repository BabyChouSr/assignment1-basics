import argparse
import json

def find_longest_token(vocab_file):
    """
    Load a vocabulary file and find the longest token in the dictionary values.
    
    Args:
        vocab_file (str): Path to the JSON vocabulary file
        
    Returns:
        tuple: (longest_token, token_length, token_id)
    """
    try:
        # Load the vocabulary file
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Initialize variables to track the longest token
        longest_token = ""
        longest_length = 0
        longest_id = None
        
        # Iterate through the vocabulary
        for token_id, token in vocab.items():
            token_length = len(token)
            if token_length > longest_length:
                longest_length = token_length
                longest_token = token
                longest_id = token_id
        
        return (longest_token, longest_length, longest_id)
    
    except FileNotFoundError:
        print(f"Error: File '{vocab_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{vocab_file}' is not a valid JSON file.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find the longest token in a vocabulary file')
    parser.add_argument('--vocab-file', type=str, default="results/TinyStoriesV2-GPT4-train-vocab.json",
                        help='Path to the JSON vocabulary file')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information about the longest token')
    
    args = parser.parse_args()
    
    result = find_longest_token(args.vocab_file)
    
    if result:
        token, length, token_id = result
        if args.verbose:
            print(f"Longest token: '{token}'")
            print(f"Length: {length}")
            print(f"Token ID: {token_id}")
        else:
            print(result)
