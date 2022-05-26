from typing import Any, TypeAlias, TypedDict
import pandas as pd
import enum

ThreeAxis: TypeAlias = tuple[float, float, float]


class WristSample(TypedDict):
    palm_gyro: ThreeAxis
    palm_acc: ThreeAxis
    arm_gyro: ThreeAxis
    arm_acc: ThreeAxis


class CalibrationData(TypedDict):
    palm_z: Any
    palm_mount: Any
    arm_z: Any
    arm_mount: Any


class DominantHand(enum.Enum):
    LEFT = 0
    RIGHT = 1


class WornHand(enum.Enum):
    DOMINANT = 0
    OFFHAND = 1


class SwingType(enum.Enum):
    FULL_SWING = 0
    PUTTING = 1


class SwingDataInstance(TypedDict):
    arm_gyro_x: pd.Series
    arm_gyro_y: pd.Series
    arm_gyro_z: pd.Series
    arm_acc_x: pd.Series
    arm_acc_y: pd.Series
    arm_acc_z: pd.Series
    palm_gyro_x: pd.Series
    palm_gyro_y: pd.Series
    palm_gyro_z: pd.Series
    palm_acc_x: pd.Series
    palm_acc_y: pd.Series
    palm_acc_z: pd.Series
    swing_type: SwingType
    dominant_hand: DominantHand
    worn_hand: WornHand
    class_id: int  # For class descriptions see class_ids.txt


def wristSample2swingDataInstance(samples: list[WristSample],
                                  swingType: SwingType,
                                  dominantHand: DominantHand,
                                  wornHand: WornHand) -> SwingDataInstance:
    instance = SwingDataInstance()
    instance["swing_type"] = swingType
    instance["dominant_hand"] = dominantHand
    instance["worn_hand"] = wornHand
    if swingType is None:
        instance["class_id"] = 0
    else:
        instance["class_id"] = 1 + dominantHand.value + (wornHand.value*2) + (swingType.value*4)
    print(f"Created SDI metadata: {instance}")
    instance["arm_gyro_x"] = pd.Series(data=[x["arm_gyro"][0] for x in samples])
    instance["arm_gyro_y"] = pd.Series(data=[x["arm_gyro"][1] for x in samples])
    instance["arm_gyro_z"] = pd.Series(data=[x["arm_gyro"][2] for x in samples])
    instance["arm_acc_x"] = pd.Series(data=[x["arm_acc"][0] for x in samples])
    instance["arm_acc_y"] = pd.Series(data=[x["arm_acc"][1] for x in samples])
    instance["arm_acc_z"] = pd.Series(data=[x["arm_acc"][2] for x in samples])
    instance["palm_gyro_x"] = pd.Series(data=[x["palm_gyro"][0] for x in samples])
    instance["palm_gyro_y"] = pd.Series(data=[x["palm_gyro"][1] for x in samples])
    instance["palm_gyro_z"] = pd.Series(data=[x["palm_gyro"][2] for x in samples])
    instance["palm_acc_x"] = pd.Series(data=[x["palm_acc"][0] for x in samples])
    instance["palm_acc_y"] = pd.Series(data=[x["palm_acc"][1] for x in samples])
    instance["palm_acc_z"] = pd.Series(data=[x["palm_acc"][2] for x in samples])
    return instance

def swingDataInstance2wristSample(sdi: SwingDataInstance):
    samples: list[WristSample] = []
    for i in range(len(sdi['arm_acc_x'])):
        s = WristSample()
        s['arm_gyro'] = (sdi['arm_gyro_x'][i], sdi['arm_gyro_y'][i], sdi['arm_gyro_z'][i],)
        s['arm_acc'] = (sdi['arm_acc_x'][i], sdi['arm_acc_y'][i], sdi['arm_acc_z'][i],)
        s['palm_gyro'] = (sdi['palm_gyro_x'][i], sdi['palm_gyro_y'][i], sdi['palm_gyro_z'][i],)
        s['palm_acc'] = (sdi['palm_acc_x'][i], sdi['palm_acc_y'][i], sdi['palm_acc_z'][i],)
        samples.append(s)
    return samples


def wristSample2sktimeData(samples: list[WristSample]) -> pd.DataFrame:
    data = {
        "arm_gyro_x": [pd.Series(data=[x["arm_gyro"][0] for x in samples], dtype='float64')],
        "arm_gyro_y": [pd.Series(data=[x["arm_gyro"][1] for x in samples], dtype='float64')],
        "arm_gyro_z": [pd.Series(data=[x["arm_gyro"][2] for x in samples], dtype='float64')],
        "arm_acc_x": [pd.Series(data=[x["arm_acc"][0] for x in samples], dtype='float64')],
        "arm_acc_y": [pd.Series(data=[x["arm_acc"][1] for x in samples], dtype='float64')],
        "arm_acc_z": [pd.Series(data=[x["arm_acc"][2] for x in samples], dtype='float64')],
        "palm_gyro_x": [pd.Series(data=[x["palm_gyro"][0] for x in samples], dtype='float64')],
        "palm_gyro_y": [pd.Series(data=[x["palm_gyro"][1] for x in samples], dtype='float64')],
        "palm_gyro_z": [pd.Series(data=[x["palm_gyro"][2] for x in samples], dtype='float64')],
        "palm_acc_x": [pd.Series(data=[x["palm_acc"][0] for x in samples], dtype='float64')],
        "palm_acc_y": [pd.Series(data=[x["palm_acc"][1] for x in samples], dtype='float64')],
        "palm_acc_z": [pd.Series(data=[x["palm_acc"][2] for x in samples], dtype='float64')]
    }
    return pd.DataFrame(data)
