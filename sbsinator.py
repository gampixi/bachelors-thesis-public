import csv

from e2e_detectors import E2EDetector, E2ERocketFullSwingPrime, E2EMinigolf, E2ERocketPuttingPrime
from minigolf import MinigolfDetector
from sbs import SBSRunner, get_d1_additonal_rate, get_d2_additonal_rate, get_misaligned_rate, SBSRunnerResult
from sensor_data_types import DominantHand, WornHand
from swing_data_instance import ArmGyroNormSynth, PalmGyroNormSynth


def get_full_swing_rocket(split: str) -> E2EDetector:
    nx = f"ROCKET Pilna Vēziena {split}"
    return E2ERocketFullSwingPrime(
        name=nx,
        crop=slice(150, 250),
        window_size=80,
        split=split
    )


def get_putting_rocket(split: str) -> E2EDetector:
    nx = f"ROCKET Ripināšana {split}"
    return E2ERocketPuttingPrime(
        name=nx,
        crop=slice(100, 225),
        dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
                              "palm_acc_x", "palm_acc_y", "palm_acc_z"],
        synthesize_dimensions=[ArmGyroNormSynth(), PalmGyroNormSynth()],
        window_size=50,
        split=split
    )


def get_full_swing_minigolf() -> E2EDetector:
    return E2EMinigolf(DominantHand.RIGHT, WornHand.OFFHAND, MinigolfDetector.FULLSWING, name="Etalons Pilna Vēziena")


def get_putting_minigolf() -> E2EDetector:
    return E2EMinigolf(DominantHand.RIGHT, WornHand.OFFHAND, MinigolfDetector.PUTTING, name="Etalons Ripināšana")


def get_splits() -> (list[str], list[str],):
    spl: list[str] = []
    spl_readable: list[str] = []
    split_sizes = [20, 500]
    split_folds = [5, 5]
    for idx, size in enumerate(split_sizes):
        for fold in range(split_folds[idx]):
            spl.append(f"split_final_{size}_F{fold}.pck")
            spl_readable.append(f"{size}-{fold}")
    return spl, spl_readable


def save_results(writer, id: int, runner: SBSRunner, result: SBSRunnerResult):
    for idx, r in enumerate(result['record_results']):
        writer.writerow([
            f"{id}", runner.name, result['detector1_name'], result['detector2_name'], result['dataset_name'],
            r['filename'],
            ";".join([f"{x}" for x in r['detector1']]),
            ";".join([f"{x}" for x in r['detector2']]),
            "1" if r['detector1_has_additional'] else 0,
            "1" if r['detector2_has_additional'] else 0,
            "1" if r['has_misaligned'] else 0,
        ])


if __name__ == "__main__":
    # Initialize CSV for results
    csv_name = "sbs_results.csv"
    csv_file = open(csv_name, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(["ID", "Izpild", "Nosaukums 1", "Nosaukums 2", "Veziena tips", "Ier",
                         "Det 1", "Det 2", "1 vairak neka 2", "2 vairak neka 1", "Nesakrit"])

    runners = []
    splits, splits_readable = get_splits()
    for s_idx, s in enumerate(splits):
        runners.append(SBSRunner(
            name=f"Pilna vēziena {splits_readable[s_idx]}",
            dataset="fs_right_off",
            detector_1_builder=get_full_swing_rocket,
            detector_2_builder=get_full_swing_minigolf,
            detector_1_args={"split": s},
            detector_2_args={}
        ))
        runners.append(SBSRunner(
            name=f"Ripināšana {splits_readable[s_idx]}",
            dataset="put_right_off",
            detector_1_builder=get_putting_rocket,
            detector_2_builder=get_putting_minigolf,
            detector_1_args={"split": s},
            detector_2_args={}
        ))

    for idx, r in enumerate(runners):
        print(f"--- RUNNING {r.name} {idx}/{len(runners)}---")
        result = r.run()
        print("Saving results...")
        save_results(csv_writer, idx, r, result)

        print(f"--- RESULTS {r.name} ---")
        print(f"1. detektoram vairāk par 2. - {get_d1_additonal_rate(result) * 100:.1f}%")
        print(f"2. detektoram vairāk par 1. - {get_d2_additonal_rate(result) * 100:.1f}%")
        print(f"Nesakrīt pozīcijas - {get_misaligned_rate(result) * 100:.1f}%")
