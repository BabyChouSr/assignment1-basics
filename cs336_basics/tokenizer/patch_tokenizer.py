"""Previously had some failures with utf-8 decoding issues when serializing, patch those.
This script patches the tokenizer vocabulary to ensure proper encoding of byte tokens.
"""

import argparse
import json
import os

def patch_tokenizer_vocab(vocab_path):
    # Load the vocabulary
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Patch the byte tokens (0-255) to use latin-1 encoding
    patched = False
    for i in range(256):
        byte_token = bytes([i]).decode('latin-1')
        if str(i) in vocab and vocab[str(i)] != byte_token:
            vocab[str(i)] = byte_token
            patched = True
    
    if patched:
        # Create backup of original file
        backup_path = vocab_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(vocab_path, backup_path)
            print(f"Created backup at {backup_path}")
        
        # Save the patched vocabulary
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"Successfully patched tokenizer vocabulary at {vocab_path}")
    else:
        print("No changes needed to the vocabulary file.")

def main(args):
    patch_tokenizer_vocab(args.vocab_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch tokenizer vocabulary for proper byte encoding")
    parser.add_argument("--vocab-path", required=True, help="Path to the tokenizer vocabulary JSON file")
    args = parser.parse_args()

    main(args)
