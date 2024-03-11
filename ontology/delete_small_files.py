# Script that deletes recursively all files in a directory that are smaller than a given size

import os
import sys
import shutil

def delete_small_files(directory, size):
    n_deleted = 0
    n_kept = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            if os.path.getsize(path) < size:
                os.remove(path)
                print(f"Deleted {path}")
                n_deleted += 1
            else:
                print(f"Kept {path}")
                n_kept += 1
    print(f"Deleted {n_deleted} files, kept {n_kept} files")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} directory size")
        sys.exit(1)
    directory = sys.argv[1]
    size = int(sys.argv[2])
    delete_small_files(directory, size)
