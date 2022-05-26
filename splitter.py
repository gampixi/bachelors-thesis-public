import random
from typing import TypedDict
from pathlib import Path
import pickle
import hashlib


class DataSplit(TypedDict):
    train: list[str]
    test: list[str]


def train_split(l: list, test_fraction: float) -> list:
    if len(l) <= 0:
        return []
    split_idx = int(len(l) * test_fraction)
    return l[split_idx:]


def test_split(l: list, test_fraction: float) -> list:
    if len(l) <= 0:
        return []
    split_idx = int(len(l) * test_fraction)
    return l[:split_idx]


# Generates even splits for all class categories
def generate_split(test_fraction: float = 0.95) -> str:
    print(f"Generating split for all data with test_fraction {test_fraction:.3f}")

    not_files: list[str] = []
    fs_right_off: list[str] = []
    fs_right_dm: list[str] = []
    fs_left_off: list[str] = []
    fs_left_dm: list[str] = []
    put_right_off: list[str] = []
    put_right_dm: list[str] = []
    put_left_off: list[str] = []
    put_left_dm: list[str] = []

    for path in Path('dataset/not').rglob('*.pck'):
        not_files.append(str(path))
    for path in Path('dataset/fs_right_off').rglob('*.pck'):
        fs_right_off.append(str(path))
    for path in Path('dataset/fs_right_dm').rglob('*.pck'):
        fs_right_dm.append(str(path))
    for path in Path('dataset/fs_left_off').rglob('*.pck'):
        fs_left_off.append(str(path))
    for path in Path('dataset/fs_left_dm').rglob('*.pck'):
        fs_left_dm.append(str(path))
    for path in Path('dataset/put_right_off').rglob('*.pck'):
        put_right_off.append(str(path))
    for path in Path('dataset/put_right_dm').rglob('*.pck'):
        put_right_dm.append(str(path))
    for path in Path('dataset/put_left_off').rglob('*.pck'):
        put_left_off.append(str(path))
    for path in Path('dataset/put_left_dm').rglob('*.pck'):
        put_left_dm.append(str(path))

    random.shuffle(not_files)
    random.shuffle(fs_right_off)
    random.shuffle(fs_right_dm)
    random.shuffle(fs_left_off)
    random.shuffle(fs_left_dm)
    random.shuffle(put_right_off)
    random.shuffle(put_right_dm)
    random.shuffle(put_left_off)
    random.shuffle(put_left_dm)

    train_files: list[str] = []
    test_files: list[str] = []

    train_files += train_split(not_files, test_fraction)
    test_files += test_split(not_files, test_fraction)

    train_files += train_split(fs_right_off, test_fraction)
    test_files += test_split(fs_right_off, test_fraction)

    train_files += train_split(fs_right_dm, test_fraction)
    test_files += test_split(fs_right_dm, test_fraction)

    train_files += train_split(fs_left_off, test_fraction)
    test_files += test_split(fs_left_off, test_fraction)

    train_files += train_split(fs_left_dm, test_fraction)
    test_files += test_split(fs_left_dm, test_fraction)

    train_files += train_split(put_right_off, test_fraction)
    test_files += test_split(put_right_off, test_fraction)

    train_files += train_split(put_right_dm, test_fraction)
    test_files += test_split(put_right_dm, test_fraction)

    train_files += train_split(put_left_off, test_fraction)
    test_files += test_split(put_left_off, test_fraction)

    train_files += train_split(put_left_dm, test_fraction)
    test_files += test_split(put_left_dm, test_fraction)

    split = DataSplit()
    split['train'] = train_files
    split['test'] = test_files

    pickled = pickle.dumps(split, fix_imports=False)
    md5_hash = hashlib.md5(pickled).hexdigest()
    filename = f"split_{test_fraction:.3f}_{md5_hash}.pck"
    print(f"Saving split to {filename}...")
    with open(filename, "wb") as file:
        file.write(pickled)
    print(f"Saved split to {filename}!")
    return filename


def load_split(filename: str) -> DataSplit:
    return pickle.load(open(filename, 'rb'))


if __name__ == "__main__":
    new_file_name = generate_split()
    print(f"Loaded from pickle: {load_split(new_file_name)}")