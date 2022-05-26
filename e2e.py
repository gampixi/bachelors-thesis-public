import glob
import os
import pickle
from typing import TypedDict

import sensor_data_reader as sr
from e2e_detectors import E2ESwingMetadata
from sensor_data_types import WristSample, CalibrationData


class DetectionDataRecord(TypedDict):
    data_path: str
    calibration_path: str
    detection_path: str


class E2ERecordResult(TypedDict):
    expected: list[int]
    got: list[int]
    false_positives: int
    true_positives: int
    false_negatives: int


class E2ERunnerResult(TypedDict):
    detector_name: str
    dataset_name: str
    record_results: list[E2ERecordResult]
    total_fp: int
    total_tp: int
    total_fn: int


def result2score(result: E2ERunnerResult) -> float:
    all_scores = []

    for r in result['record_results']:
        false_positives = r['false_positives']
        true_positives = r['true_positives']
        false_negatives = r['false_negatives']

        total = false_positives + true_positives + false_negatives
        if total == 0:
            if len(r['expected']) == 0:
                score = 1.0
            else:
                score = 0.0
        else:
            score = true_positives / (false_positives + true_positives + false_negatives)

        all_scores.append(score)
    return sum(all_scores) / len(all_scores)


def result2precision(result: E2ERunnerResult) -> float:
    return result['total_tp'] / (result['total_tp'] + result['total_fp'])


def result2recall(result: E2ERunnerResult) -> float:
    total_expected = 0
    for r in result['record_results']:
        total_expected += len(r['expected'])
    return result['total_tp'] / total_expected


def load_raw_dataset(dataset: str) -> list[DetectionDataRecord]:
    ret: list[DetectionDataRecord] = []
    path_to = f"./e2e_dataset/{dataset}/"
    bin_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.bin")]
    cal_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.cal")]
    for bix, b in enumerate(bin_files):
        common_part = b[0:-4]
        matching_cal_name = f"{common_part}_CD.cal"
        matching_pck_name = f"{common_part}.pck"
        for idx, c in enumerate(cal_files):
            if c == matching_cal_name:
                ret.append({
                    "data_path": f"{path_to}{b}",
                    "calibration_path": f"{path_to}{c}",
                    "detection_path": f"{path_to}{matching_pck_name}"
                })
                bin_files[bix] = "USED UP NAME"
                cal_files[idx] = "USED UP NAME"
                break
    return ret


def get_samples(filename) -> list[WristSample]:
    with open(filename, 'rb') as file:
        return sr.binparse(file.read())


def get_calibration(filename) -> CalibrationData:
    with open(filename, 'rb') as file:
        return sr.calparse(file.read())


def load_data(data_path: str, cal_path: str):
    samples = get_samples(data_path)
    calibration = get_calibration(cal_path)
    sr.apply_calibration(samples, calibration)
    return samples


def load_detections(detection_path) -> list[int]:
    if not os.path.exists(detection_path):
        return []
    loaded: E2ESwingMetadata = pickle.load(open(detection_path, 'rb'))
    return loaded["impact_positions"]
