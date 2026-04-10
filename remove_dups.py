import os
import hashlib

def clean_duplicates(folder):
    seen_hashes = set()
    removed_count = 0
    kept_count = 0
    
    # We walk the directory tree
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            try:
                with open(p, 'rb') as file_obj:
                    # Read the entire file to get an MD5 hash
                    file_hash = hashlib.md5(file_obj.read()).hexdigest()
                    
                if file_hash in seen_hashes:
                    # We have seen this identical file before. Delete the copy.
                    os.remove(p)
                    removed_count += 1
                else:
                    # It's the first time we see this file. Keep it and record the hash.
                    seen_hashes.add(file_hash)
                    kept_count += 1
            except Exception as e:
                print(f"Error processing {p}: {e}")
                
    print(f"--- Deduplication Summary ---")
    print(f"Target Directory: {folder}")
    print(f"Kept: {kept_count} strictly unique files.")
    print(f"Deleted: {removed_count} duplicate copies.")

if __name__ == "__main__":
    target_folder = r"c:\Users\sanga\Downloads\SATHWIK\Documents\src\data\raw\dataset-11"
    clean_duplicates(target_folder)
