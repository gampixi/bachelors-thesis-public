from matplotlib import pyplot as plt

from e2e import E2ERecordResult, E2ERunnerResult, load_raw_dataset, load_data, load_detections, result2precision, \
    result2recall
from e2e_detectors import E2EDetector
from typing import Callable, Any

from visualization import plot_samples


class E2ERunner:
    def __init__(self, name: str, dataset: str,
                 detector_builder: Callable[[], E2EDetector] | None = None,
                 detector_builder_2: Callable[[Any], E2EDetector] | None = None,
                 db2_args: dict = {},
                 impact_dilation: int = 7):
        main_dataset = load_raw_dataset(dataset)
        not_dataset = load_raw_dataset("not")
        self.name = name
        self.dataset_name = dataset
        self.dataset = main_dataset + not_dataset
        self.detector_builder = detector_builder
        self.detector_builder_2 = detector_builder_2
        self.db2_args = db2_args
        self.impact_dilation = impact_dilation

    def run(self) -> E2ERunnerResult:
        db = None
        if self.detector_builder is not None:
            db = self.detector_builder
        if self.detector_builder_2 is not None:
            db = lambda a=self.db2_args: self.detector_builder_2(**a)

        records = self.dataset
        expected_detections = [load_detections(r["detection_path"]) for r in records]
        all_detections = [[] for x in records]
        final_result: E2ERunnerResult = E2ERunnerResult(
            detector_name=db().get_name(),
            dataset_name=self.dataset_name,
            record_results=[],
            total_fn=0,
            total_fp=0,
            total_tp=0
        )

        for idx, rdr in enumerate(records):
            # Initialize the detector anew for each run
            filehash = (rdr["data_path"].split('/')[-1]).split('_')[0]
            # print(f"Evaluating {filehash}...")
            detector: E2EDetector = db()
            samples = load_data(rdr["data_path"], rdr["calibration_path"])
            for sample in samples:
                result = detector.add_sample(sample)

                if result is not None:
                    # print(f"Detected @ {result}")
                    all_detections[idx].append(result)

            expected = expected_detections[idx]
            got = all_detections[idx]

            false_positives = 0
            fp_pos = []
            true_positives = 0
            tp_pos = []
            false_negatives = 0
            fn_pos = []

            for d in got:
                dilated = [x for x in range(d - self.impact_dilation, d + self.impact_dilation + 1)]
                if any(i in dilated for i in expected):
                    # Was in expected and was in got
                    true_positives += 1
                    # Remove the expected value once it's been matched to prevent double
                    # detections from both counting as a true positive
                    expected = list(filter(lambda x: x not in dilated, expected))
                    tp_pos.append(d)
                else:
                    # Was not expected, was got
                    false_positives += 1
                    fp_pos.append(d)

            for e in expected:
                dilated = [x for x in range(e - self.impact_dilation, e + self.impact_dilation + 1)]
                if not any(i in dilated for i in got):
                    # Was in expected, but not in got
                    false_negatives += 1
                    fn_pos.append(e)

            for fp in fp_pos:
                start_idx = fp - 200
                end_idx = fp + 100
                if start_idx < 0:
                    start_idx = 0
                if end_idx >= len(samples):
                    end_idx = len(samples) - 1
                det_markers = [fp - start_idx]
                t_markers = [p - start_idx for p in expected_detections[idx]]
                fig, ax = plot_samples(samples[start_idx:end_idx], det_markers, t_markers)
                ds = rdr["data_path"].split('/')[-2]
                fig.suptitle(f"{ds} {filehash} {expected} {got}", fontsize=14)
                fig.savefig(f"FP-{self.name}-{filehash}-{fp}.png", dpi=100)
                plt.close(fig)

            for fp in fn_pos:
                start_idx = fp - 200
                end_idx = fp + 100
                if start_idx < 0:
                    start_idx = 0
                if end_idx >= len(samples):
                    end_idx = len(samples) - 1
                det_markers = [p - start_idx for p in got]
                t_markers = [p - start_idx for p in expected_detections[idx]]
                fig, ax = plot_samples(samples[start_idx:end_idx], det_markers, t_markers)
                ds = rdr["data_path"].split('/')[-2]
                fig.suptitle(f"{ds} {filehash} {expected} {got}", fontsize=14)
                fig.savefig(f"FN-{self.name}-{filehash}-{fp}.png", dpi=100)
                plt.close(fig)

            final_result["total_tp"] += true_positives
            final_result["total_fp"] += false_positives
            final_result["total_fn"] += false_negatives
            final_result["record_results"].append(E2ERecordResult(
                expected=expected_detections[idx],
                got=all_detections[idx],
                false_positives=false_positives,
                false_negatives=false_negatives,
                true_positives=true_positives
            ))

        return final_result


def print_results(res: E2ERunnerResult):
    precision = result2precision(res)
    recall = result2recall(res)
    print(f"{res['detector_name']} -- {res['dataset_name']}")
    print(f"Total true positives: {res['total_tp']}")
    print(f"Total false positives: {res['total_fp']}")
    print(f"Total false negatives: {res['total_fn']}")
    print(f"Precision: {(precision * 100):.1f}%\t Recall: {(recall * 100):.1f}%")
    print("---------------------------------")


def save_results(writer, id: int, runner: E2ERunner, res: E2ERunnerResult):
    writer.writerow([f"{id}", runner.name, runner.dataset_name, res['total_tp'],
                     res['total_fp'], res['total_fn'], result2precision(res), result2recall(res)])


def execute_runner(id: int, r: E2ERunner) -> (int, E2ERunner, E2ERunnerResult):
    print(f"Running {r.name}")
    res = r.run()
    print_results(res)
    return id, r, res