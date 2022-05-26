import typing
from typing import Callable
from sensor_data_types import *
from splitter import *
import pickle
import hashlib
import pandas as pd
import numpy as np

def sdi_save(sdi: SwingDataInstance) -> str:
    base_path = "dataset/"
    subdirectory_parts = []
    if sdi['swing_type'] is None:
        subdirectory_parts.append("not")
    else:
        match sdi['swing_type']:
            case SwingType.FULL_SWING:
                subdirectory_parts.append("fs")
            case SwingType.PUTTING:
                subdirectory_parts.append("put")
        match sdi['dominant_hand']:
            case DominantHand.RIGHT:
                subdirectory_parts.append("right")
            case DominantHand.LEFT:
                subdirectory_parts.append("left")
        match sdi['worn_hand']:
            case WornHand.OFFHAND:
                subdirectory_parts.append("off")
            case WornHand.DOMINANT:
                subdirectory_parts.append("dm")

    subdirectory = f"{'_'.join(subdirectory_parts)}/"
    pickled = pickle.dumps(sdi, fix_imports=False)
    md5_hash = hashlib.md5(pickled).hexdigest()
    filename = f"{base_path}{subdirectory}{'_'.join(subdirectory_parts)}_{md5_hash}.pck"
    print(f"Saving SDI to {filename}...")
    with open(filename, "wb") as file:
        file.write(pickled)
    print(f"Saved SDI to {filename}!")
    return filename


def sdi_load(filename: str) -> SwingDataInstance:
    return pickle.load(open(filename, 'rb'))


def sdi_load_split(filename: str) -> (list[SwingDataInstance], list[SwingDataInstance],):
    split_data = load_split(filename)
    print("Loading training data...")
    train_sdi: list[SwingDataInstance] = [sdi_load(x) for x in split_data['train']]
    print("Loading testing data...")
    test_sdi: list[SwingDataInstance] = [sdi_load(x) for x in split_data['test']]
    return train_sdi, test_sdi


def sdiList2sktimeData(sdil: list[SwingDataInstance]) -> (pd.DataFrame, list[int]):
    data = {
        "arm_gyro_x": [],
        "arm_gyro_y": [],
        "arm_gyro_z": [],
        "arm_acc_x": [],
        "arm_acc_y": [],
        "arm_acc_z": [],
        "palm_gyro_x": [],
        "palm_gyro_y": [],
        "palm_gyro_z": [],
        "palm_acc_x": [],
        "palm_acc_y": [],
        "palm_acc_z": []
    }
    classes: list[int] = []
    print("Generating dictionary...")
    for sdi in sdil:
        for key in data.keys():
            data[key].append(sdi[key])
        classes.append(sdi['class_id'])
    print("Dictionary to DataFrame")
    df = pd.DataFrame(data)
    return df, classes


class DimensionSynth:
    def get_name(self) -> str:
        pass

    def get_series(self, src) -> pd.Series:
        pass


class PalmGyroNormSynth(DimensionSynth):
    def __str__(self):
        return self.get_name()

    def get_name(self) -> str:
        return "palm_gyro_norm"

    def get_series(self, src) -> pd.Series:
        palm_x: pd.Series = src.palm_gyro_x
        palm_y: pd.Series = src.palm_gyro_y
        palm_z: pd.Series = src.palm_gyro_z

        values = []
        for i in range(len(palm_x)):  # Assume all series are of equal length (if they aren't that's bad!)
            values.append(np.linalg.norm([palm_x[i], palm_y[i], palm_z[i]]))

        return pd.Series(data=values, dtype='float64')


class ArmGyroNormSynth(DimensionSynth):
    def __str__(self):
        return self.get_name()

    def get_name(self) -> str:
        return "arm_gyro_norm"

    def get_series(self, src) -> pd.Series:
        arm_x: pd.Series = src.arm_gyro_x
        arm_y: pd.Series = src.arm_gyro_y
        arm_z: pd.Series = src.arm_gyro_z

        values = []
        for i in range(len(arm_x)):  # Assume all series are of equal length (if they aren't that's bad!)
            values.append(np.linalg.norm([arm_x[i], arm_y[i], arm_z[i]]))

        return pd.Series(data=values, dtype='float64')


class PalmAccDifSynth(DimensionSynth):
    def __str__(self):
        return self.get_name()

    def get_name(self) -> str:
        return "arm_acc_dif"

    def get_series(self, src) -> pd.Series:
        palm_x: pd.Series = src.palm_acc_x
        palm_y: pd.Series = src.palm_acc_y
        palm_z: pd.Series = src.palm_acc_z

        values = []
        last_acc = None
        for i in range(len(palm_x)):  # Assume all series are of equal length (if they aren't that's bad!)
            if last_acc is not None:
                values.append(np.linalg.norm(
                    [palm_x[i] - last_acc[0],
                     palm_y[i] - last_acc[1],
                     palm_z[i] - last_acc[2]]))

            if len(values) == 1:
                # Duplicate the first to fix 1 sample offset
                values.append(values[0])
            last_acc = [palm_x[i], palm_y[i], palm_z[i]]

        return pd.Series(data=values, dtype='float64')


def skd_post_process(df: pd.DataFrame, cl: list[int],
                     crop_series_rows: slice = None,
                     synthesize_dimensions: list[DimensionSynth] = [],
                     classes_to_remove: list[int] = [],
                     class_remap: typing.Mapping[int, int] = {},
                     dimensions_to_remove: list[str] = []) -> pd.DataFrame:
    cf = df

    # Remap classes in class list
    for idx, c in enumerate(cl):
        if c in class_remap.keys():
            cl[idx] = class_remap[c]

    # Remove specified classes from dataset
    # 1. Find indexes on the classes
    idx_to_remove = []
    for idx, c in enumerate(cl):
        if c in classes_to_remove:
            idx_to_remove.append(idx)

    # 2. Remove the rows from DataFrame
    cf = cf.drop(idx_to_remove, axis=0)

    # 3. Remove the rows from class list
    idx_to_remove.sort(reverse=True)  # Start removing from back indexes so the changes in order have no impact
    for idx in idx_to_remove:
        cl.pop(idx)

    # Apply slicing if provided
    if crop_series_rows is not None:
        # Unoptimized shit because a pd.Series slice retains indexing, but we need it to start from 0
        cf = cf.applymap(lambda x: pd.Series(data=x.tolist()[crop_series_rows]))

    # Synthesize dimensions
    # Do it now so we don't calculate them for unnecessary rows
    for d in synthesize_dimensions:
        new_column_name = d.get_name()
        new_column = []
        for r in cf.itertuples(index=False):
            new_column.append(d.get_series(r))
        cf.insert(0, new_column_name, new_column)

    # Remove specified dimensions
    cf = cf.drop(dimensions_to_remove, axis=1)

    return cf
