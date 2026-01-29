import os
import shutil
import random
import argparse
from pathlib import Path

def prepare_data(input_dir, output_dir, split_ratio=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    all_files = [f for f in input_path.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    random.shuffle(all_files)
    
    total_files = len(all_files)
    train_count = int(total_files * split_ratio[0])
    val_count = int(total_files * split_ratio[1])
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    
    print(f"Total images: {total_files}")
    print(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    def copy_files(files, dest_subdir):
        for f in files:
            shutil.copy2(f, output_path / dest_subdir / f.name)
            
    print("Copying train files...")
    copy_files(train_files, 'train')
    
    print("Copying val files...")
    copy_files(val_files, 'val')
    
    print("Copying test files...")
    copy_files(test_files, 'test')
    
    print("Data preparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed data")
    args = parser.parse_args()
    
    prepare_data(args.input, args.output)
