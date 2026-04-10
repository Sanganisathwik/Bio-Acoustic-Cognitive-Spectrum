import os
import hashlib
from collections import defaultdict
import sys

def get_duplicates(folder):
    size_dict = defaultdict(list)
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            try:
                size = os.path.getsize(p)
                size_dict[size].append(p)
            except OSError:
                pass
    
    hash_dict = defaultdict(list)
    for size, paths in size_dict.items():
        if len(paths) > 1:
            for p in paths:
                try:
                    with open(p, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    hash_dict[file_hash].append(p)
                except Exception:
                    pass
    
    duplicates = [paths for paths in hash_dict.values() if len(paths) > 1]
    return duplicates

def main():
    dirs_to_check = [
        r"c:\Users\sanga\Downloads\SATHWIK\Documents\src\data\raw\dataset-11",
        r"c:\Users\sanga\Downloads\SATHWIK\Documents\src\data\bio",
        r"c:\Users\sanga\Downloads\SATHWIK\Documents\src\data\nonbio"
    ]
    
    for d in dirs_to_check:
        print(f"\n--- Checking for duplicates in {os.path.basename(d)} ---")
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            continue
            
        dups = get_duplicates(d)
        if dups:
            num_duplicate_files = sum(len(group) for group in dups) - len(dups)
            print(f"Found {num_duplicate_files} exact duplicates (based on file content hash):")
            for i, group in enumerate(dups, 1):
                print(f"  Group {i}:")
                for path in group:
                    print(f"    - {os.path.relpath(path, start=r'c:\Users\sanga\Downloads\SATHWIK\Documents\src')}")
        else:
            print("No exact duplicates found.")

if __name__ == "__main__":
    main()
