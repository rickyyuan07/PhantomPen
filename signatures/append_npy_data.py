import os
import numpy as np
import argparse
import shutil
import glob

import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy npy files in source directory to a (non-empty) target directory")
    parser.add_argument("-s", "--source_dir", type=str, required=True, help="source directory")
    parser.add_argument("-t", "--target_dir", type=str, required=True, help="target directory")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    target_files = [f for f in os.listdir(args.target_dir) if f.endswith(".npy")]
    args.signature_idx = 0
    if target_files:
        args.signature_idx = max([int(os.path.splitext(f)[0]) for f in target_files]) + 1

    source_files = [f for f in os.listdir(args.source_dir) if f.endswith(".npy")]
    new_files = [os.path.join(args.target_dir, f"{f}.npy") for f in range(args.signature_idx, args.signature_idx + len(source_files))]
    for s, t in zip(source_files, new_files):
        shutil.copy(os.path.join(args.source_dir, s), t)

    
    

