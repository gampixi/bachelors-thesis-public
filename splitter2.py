import pickle
import sys
from pathlib import Path
from splitter import DataSplit
import random


def test_split(l: list, test_fraction: float) -> list:
    if len(l) <= 0:
        return []
    split_idx = int(len(l) * test_fraction)
    return l[:split_idx]


def max_split(l: list, max_items: int) -> list:
    try:
        r = random.sample(l, max_items)
        return r
    except ValueError:
        return l


def generate_split(max_items: int) -> DataSplit:
    not_files: list[str] = []
    fs_right_off: list[str] = []
    put_right_off: list[str] = []

    for path in Path('dataset/not').rglob('*.pck'):
        not_files.append(str(path))
    for path in Path('dataset/fs_right_off').rglob('*.pck'):
        fs_right_off.append(str(path))
    for path in Path('dataset/put_right_off').rglob('*.pck'):
        put_right_off.append(str(path))

    test_files = []
    train_files = []
    test_files += test_split(not_files, 0.2)
    test_files += test_split(fs_right_off, 0.2)
    test_files += test_split(put_right_off, 0.2)
    train_files += max_split(not_files, max_items)
    train_files += max_split(fs_right_off, max_items)
    train_files += max_split(put_right_off, max_items)

    split = DataSplit(
        train=train_files,
        test=test_files
    )
    return split


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} [split size] [fold count] [name]")
        exit(1)
    split_size = int(sys.argv[1])
    folds = int(sys.argv[2])
    name = sys.argv[3]

    for i in range(folds):
        filename = f"split_{name}_{split_size}_F{i}.pck"
        print(f"Generating {filename}...")
        split = generate_split(split_size)
        print(f"Saving {filename}...")
        pickled = pickle.dumps(split, fix_imports=False)
        with open(filename, "wb") as file:
            file.write(pickled)
