import glob
from typing import Callable, Any, TypedDict

from matplotlib import pyplot as plt

from e2e import load_raw_dataset, DetectionDataRecord, load_data
from e2e_detectors import E2EDetector
from visualization import plot_samples


def load_sbs_dataset(dataset: str) -> list[DetectionDataRecord]:
    ret: list[DetectionDataRecord] = []
    path_to = f"./sbs_dataset/{dataset}/"
    bin_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.bin")]
    cal_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.cal")]
    for bix, b in enumerate(bin_files):
        common_part = b[0:-4]
        matching_cal_name = f"{common_part}_CD.cal"
        for idx, c in enumerate(cal_files):
            if c == matching_cal_name:
                ret.append({
                    "data_path": f"{path_to}{b}",
                    "calibration_path": f"{path_to}{c}",
                    "detection_path": None
                })
                bin_files[bix] = "USED UP NAME"
                cal_files[idx] = "USED UP NAME"
                break
    return ret


# Types of dissimilarity
# Additional detections
# Misaligned detections (+- 10 samples are permitted)


class SBSRecordResult(TypedDict):
    filename: str
    detector1: list[int]
    detector2: list[int]
    detector1_has_additional: bool
    detector2_has_additional: bool
    has_misaligned: bool


class SBSRunnerResult(TypedDict):
    detector1_name: str
    detector2_name: str
    dataset_name: str
    record_results: list[SBSRecordResult]


def get_d1_additonal_rate(result: SBSRunnerResult):
    cnt = 0
    for r in result['record_results']:
        if r['detector1_has_additional']:
            cnt += 1
    return cnt / len(result['record_results'])


def get_d2_additonal_rate(result: SBSRunnerResult):
    cnt = 0
    for r in result['record_results']:
        if r['detector2_has_additional']:
            cnt += 1
    return cnt / len(result['record_results'])


def get_misaligned_rate(result: SBSRunnerResult):
    cnt = 0
    for r in result['record_results']:
        if r['has_misaligned']:
            cnt += 1
    return cnt / len(result['record_results'])


class SBSRunner:
    def __init__(self, name: str, dataset: str,
                 detector_1_builder: Callable[[Any], E2EDetector],
                 detector_1_args: dict,
                 detector_2_builder: Callable[[Any], E2EDetector],
                 detector_2_args: dict):
        main_dataset = load_sbs_dataset(dataset)
        # main_e2e_dataset = load_raw_dataset(dataset)
        # not_dataset = load_raw_dataset("not")
        self.name = name
        self.dataset_name = dataset
        self.dataset = main_dataset
        # self.dataset = main_dataset + main_e2e_dataset + not_dataset
        self.detector_1_builder = detector_1_builder
        self.detector_2_builder = detector_2_builder
        self.detector_1_args = detector_1_args
        self.detector_2_args = detector_2_args

    def run(self) -> SBSRunnerResult:
        db1 = lambda a=self.detector_1_args: self.detector_1_builder(**a)
        db2 = lambda a=self.detector_2_args: self.detector_2_builder(**a)

        final_result: SBSRunnerResult = SBSRunnerResult(
            detector1_name=db1().get_name(),
            detector2_name=db2().get_name(),
            dataset_name=self.dataset_name,
            record_results=[]
        )

        records = self.dataset
        for idx, rdr in enumerate(records):
            filehash = (rdr["data_path"].split('/')[-1]).split('_')[0]

            d1: E2EDetector = db1()
            d2: E2EDetector = db2()

            res = SBSRecordResult(
                filename=filehash,
                detector1=[],
                detector2=[],
                detector1_has_additional=False,
                detector2_has_additional=False,
                has_misaligned=False
            )

            samples = load_data(rdr["data_path"], rdr["calibration_path"])
            for sample in samples:
                r1 = d1.add_sample(sample)
                r2 = d2.add_sample(sample)

                min_allowed = 200
                max_allowed = len(samples) - 200

                if r1 is not None:
                    # Since ROCKET detector can have issues triggering at the very edges
                    # Ignore edge detections for both of them to ensure fairness
                    if r1 > min_allowed and r1 < max_allowed:
                        res['detector1'].append(r1)

                if r2 is not None:
                    # Since ROCKET detector can have issues triggering at the very edges
                    # Ignore edge detections for both of them to ensure fairness
                    if r2 > min_allowed and r2 < max_allowed:
                        res['detector2'].append(r2)

            # Check for additional
            if len(res['detector1']) > len(res['detector2']):
                res['detector1_has_additional'] = True

            if len(res['detector2']) > len(res['detector1']):
                res['detector2_has_additional'] = True

            # Check for misaligned
            for x in res['detector1']:
                for y in res['detector2']:
                    srt = [x, y]
                    srt.sort()
                    delta = srt[1] - srt[0]
                    if delta > 10:
                        res['has_misaligned'] = True
                        break
                if res['has_misaligned']:
                    break

            if res['detector1_has_additional'] or res['detector2_has_additional'] or res['has_misaligned']:
                print("Saving chart...")
                # Save charts on discrepancies
                # Chart should cover area between all detections
                all_detections = res['detector1'] + res['detector2']
                all_detections.sort()
                start_idx = all_detections[0] - 200
                end_idx = all_detections[-1] + 100
                if start_idx < 0:
                    start_idx = 0
                if end_idx >= len(samples):
                    end_idx = len(samples) - 1

                d1_markers = res['detector1'].copy()
                d2_markers = res['detector2'].copy()
                for i in range(len(d1_markers)):
                    d1_markers[i] -= start_idx
                for i in range(len(d2_markers)):
                    d2_markers[i] -= start_idx

                fig, ax = plot_samples(samples[start_idx:end_idx], d1_markers, d2_markers)
                ds = rdr["data_path"].split('/')[-2]
                fig.suptitle(f"{ds} {filehash} D1A:{res['detector1_has_additional']} D2A:{res['detector2_has_additional']} MIS:{res['has_misaligned']}", fontsize=14)
                fig.savefig(f"DIS-{self.name}-{filehash}.png", dpi=100)
                plt.close(fig)

            print(f"Finished {idx}/{len(records)} {filehash} with {res}")
            final_result['record_results'].append(res)

        return final_result