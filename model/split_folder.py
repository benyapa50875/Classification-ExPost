import os
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

def create_directories(base_dir, categories):
    for category in categories:
        for subset in ['train', 'test', 'val']:
            os.makedirs(os.path.join(base_dir, category, subset), exist_ok=True)

def split_and_move_files(base_dir, category, train_ratio=0.7, test_ratio=0.2):
    category_dir = os.path.join(base_dir, category)
    files = [f for f in os.listdir(category_dir) if f.endswith('.npy')]

    train_files, test_val_files = train_test_split(files, test_size=1-train_ratio, random_state=42)
    test_files, val_files = train_test_split(test_val_files, test_size=test_ratio/(test_ratio + (1-train_ratio)), random_state=42)

    for file_set, subset in zip([train_files, test_files, val_files], ['train', 'test', 'val']):
        for file_name in file_set:
            src_path = os.path.join(category_dir, file_name)
            dest_dir = os.path.join(base_dir, category, subset)
            shutil.move(src_path, dest_dir)

if __name__ == "__main__":
    base_dir = 'datasets'
    categories = os.listdir(base_dir)

    create_directories(base_dir, categories)

    for category in categories:
        split_and_move_files(base_dir, category)
