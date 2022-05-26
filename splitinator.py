import csv
import multiprocessing

from e2e_detectors import E2ERocketFullSwingPrime, E2ERocketPuttingPrime, E2EDetector, E2ERocketFullSwingIsolation, \
    E2ERocketPuttingIsolation, E2EMinigolf, E2EThreshold
from e2e_runner import E2ERunner, save_results, execute_runner
from minigolf import MinigolfDetector
from sensor_data_types import DominantHand, WornHand
from swing_data_instance import ArmGyroNormSynth, PalmGyroNormSynth, PalmAccDifSynth


def get_full_swing_rocket_with(split: str) -> E2EDetector:
    nx = f"RFS SPLIT {split}"
    return E2ERocketFullSwingPrime(
        name=nx,
        crop=slice(125, 275),
        dimensions_to_remove=["palm_gyro_x", "palm_gyro_y", "palm_gyro_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z"],
        synthesize_dimensions=[ArmGyroNormSynth(), PalmGyroNormSynth()],
        window_size=90,
        split=split
    )


def get_putting_rocket_with(split: str) -> E2EDetector:
    nx = f"RPT SPLIT {split}"
    return E2ERocketPuttingPrime(
        name=nx,
        crop=slice(100, 225),
        dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
                              "palm_acc_x", "palm_acc_y", "palm_acc_z"],
        synthesize_dimensions=[PalmAccDifSynth()],
        window_size=30,
        split=split
    )


def get_full_swing_isolation_with(split: str) -> E2EDetector:
    nx = f"IFS SPLIT {split}"
    return E2ERocketFullSwingIsolation(
        name=nx,
        crop=slice(125, 275),
        dimensions_to_remove=["palm_gyro_x", "palm_gyro_y", "palm_gyro_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z"],
        synthesize_dimensions=[ArmGyroNormSynth(), PalmGyroNormSynth()],
        window_size=90,
        split=split
    )


def get_putting_isolation_with(split: str) -> E2EDetector:
    nx = f"IPT SPLIT {split}"
    return E2ERocketPuttingIsolation(
        name=nx,
        crop=slice(100, 225),
        dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
                              "palm_acc_x", "palm_acc_y", "palm_acc_z"],
        synthesize_dimensions=[PalmAccDifSynth()],
        window_size=30,
        split=split
    )


def get_full_swing_minigolf() -> E2EDetector:
    nx = f"MFS"
    return E2EMinigolf(
        name=nx,
        dominantHand=DominantHand.RIGHT,
        wornHand=WornHand.OFFHAND,
        detector=MinigolfDetector.FULLSWING
    )


def get_putting_minigolf() -> E2EDetector:
    nx = f"MPT"
    return E2EMinigolf(
        name=nx,
        dominantHand=DominantHand.RIGHT,
        wornHand=WornHand.OFFHAND,
        detector=MinigolfDetector.PUTTING
    )


def get_full_swing_threshold() -> E2EDetector:
    nx = f"TFS"
    return E2EThreshold(
        name=nx,
        palm_vibration_threshold=20,
        arm_gyro_x_threshold=None,
        palm_gyro_z_dif_threshold=-100,
        window_size=80,
        cooldown_period=100
    )


def get_putting_threshold() -> E2EDetector:
    nx = f"TPT"
    return E2EThreshold(
        name=nx,
        arm_gyro_x_threshold=30,
        palm_vibration_threshold=2.5,
        window_size=80,
        cooldown_period=100
    )


# Multiprocessing requires this
if __name__ == "__main__":
    ctx = multiprocessing.get_context('spawn')

    # Initialize CSV for results
    csv_name = "splitinator_results.csv"
    csv_file = open(csv_name, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(["ID", "Nosaukums", "Veziena tips", "Patiesi poz.", "Aplami poz.",
                         "Aplami neg.", "Precizitate", "Parklajums"])

    # Iniciet visus E2ERunner
    splits: list[str] = []
    splits_readable: list[str] = []
    split_sizes = [500]
    split_folds = [5]
    for idx, size in enumerate(split_sizes):
        for fold in range(split_folds[idx]):
            splits.append(f"split_final_{size}_F{fold}.pck")
            splits_readable.append(f"{size}-{fold}")

    runners: list[E2ERunner] = []

    runners.append(E2ERunner(name=f"Minigolf Full Swing", dataset="fs_right_off",
                             detector_builder_2=get_full_swing_minigolf,
                             db2_args={}))
    runners.append(E2ERunner(name=f"Minigolf Putting", dataset="put_right_off",
                             detector_builder_2=get_putting_minigolf,
                             db2_args={}))
    runners.append(E2ERunner(name=f"Threshold Full Swing", dataset="fs_right_off",
                             detector_builder_2=get_full_swing_threshold,
                             db2_args={}))
    runners.append(E2ERunner(name=f"Threshold Putting", dataset="put_right_off",
                             detector_builder_2=get_putting_threshold,
                             db2_args={}))
    # for idx, s in enumerate(splits):
    #     name = f"Dalijums {splits_readable[idx]}"
    #     runners.append(E2ERunner(name=f"R {name}", dataset="fs_right_off",
    #                              detector_builder_2=get_full_swing_rocket_with,
    #                              db2_args={"split": s}))
    #     runners.append(E2ERunner(name=f"R {name}", dataset="put_right_off",
    #                              detector_builder_2=get_putting_rocket_with,
    #                              db2_args={"split": s}))
    #     runners.append(E2ERunner(name=f"I {name}", dataset="fs_right_off",
    #                              detector_builder_2=get_full_swing_isolation_with,
    #                              db2_args={"split": s}))
    #     runners.append(E2ERunner(name=f"I {name}", dataset="put_right_off",
    #                              detector_builder_2=get_putting_isolation_with,
    #                              db2_args={"split": s}))

    # Iniciet multiprocessing
    pool = ctx.Pool(processes=4)

    # Palaist eksperimentus
    processes = [pool.apply_async(execute_runner, args=(id, r,)) for id, r in enumerate(runners)]
    results = [p.get() for p in processes]

    for r in results:
        save_results(csv_writer, r[0], r[1], r[2])

    # Beigas aizvert failu
    csv_file.close()
