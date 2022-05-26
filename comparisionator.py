# Runs a large suite of configurations and exports the results to CSV

# Full swing baseline:
# E2ERocketFullSwingPrime(name="ROCKET Mod Pilna Vēziena",
#     crop=slice(150, 250),
#     dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
#                           "palm_acc_x", "palm_acc_y", "palm_acc_z"],
#     synthesize_dimensions=[PalmAccDifSynth()])

# Putting baseline:
# E2ERocketPuttingPrime(name="ROCKET Mod Ripināšana",
#     crop=slice(100, 250),
#     dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
#                           "palm_acc_x", "palm_acc_y", "palm_acc_z",
#                           "palm_gyro_y", "palm_gyro_z",
#                           "arm_gyro_y", "arm_gyro_z"],
#     synthesize_dimensions=[PalmAccDifSynth(), ArmGyroNormSynth(), PalmGyroNormSynth()])
import multiprocessing

import csv

from e2e_detectors import E2ERocketPuttingIsolation, E2ERocketFullSwingIsolation, E2EDetector, E2ERocketFullSwingPrime, \
    E2ERocketPuttingPrime
from e2e_runner import E2ERunner, save_results, execute_runner
from swing_data_instance import PalmAccDifSynth, ArmGyroNormSynth, PalmGyroNormSynth, DimensionSynth


def get_full_swing_rocket_with(
        split: str,
        crop: slice = slice(150, 250),
        dimensions_to_remove: list[str] = [],
        synthesize_dimensions: list[DimensionSynth] = [],
        window_size: int = 50
) -> E2EDetector:
    nx = f"RFS C {crop} DR {dimensions_to_remove} SYNTH {synthesize_dimensions} WS {window_size} {split}"
    return E2ERocketFullSwingPrime(
        name=nx,
        split=split,
        crop=crop,
        dimensions_to_remove=dimensions_to_remove,
        synthesize_dimensions=synthesize_dimensions,
        window_size=window_size
    )


def get_putting_rocket_with(
        split: str,
        crop: slice = slice(100, 225),
        dimensions_to_remove: list[str] = [],
        synthesize_dimensions: list[DimensionSynth] = [],
        window_size: int = 50
) -> E2EDetector:
    nx = f"RPT C {crop} DR {dimensions_to_remove} SYNTH {synthesize_dimensions} WS {window_size} {split}"
    return E2ERocketPuttingPrime(
        name=nx,
        split=split,
        crop=crop,
        dimensions_to_remove=dimensions_to_remove,
        synthesize_dimensions=synthesize_dimensions,
        window_size=window_size
    )


all_slices = [slice(0, 300), slice(0, 100), slice(100, 200), slice(100, 225), slice(100, 250), slice(200, 300),
              slice(175, 225), slice(150, 250), slice(125, 275)]
all_dimensions_to_remove = [
    ("Tikai paatrinajums", ["palm_gyro_x", "palm_gyro_y", "palm_gyro_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z"]),
    ("Tikai ziroskops", ["palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z"]),
    ("Tikai ziroskops X", [
        "palm_gyro_y", "palm_gyro_z", "arm_gyro_y", "arm_gyro_z",
        "palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z"
    ]),
    ("Tikai ziroskops Z", [
        "palm_gyro_y", "palm_gyro_x", "arm_gyro_y", "arm_gyro_x",
        "palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z"
    ]),
    ("Tikai rokas ziroskops", [
        "palm_gyro_z", "palm_gyro_y", "palm_gyro_x",
        "palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z"
    ]),
    ("Tikai plaukstas ziroskops", [
        "arm_gyro_z", "arm_gyro_y", "arm_gyro_x",
        "palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z"
    ]),
]
all_dimensions = [
    "palm_acc_x", "palm_acc_y", "palm_acc_z", "arm_acc_x", "arm_acc_y", "arm_acc_z",
    "palm_gyro_x", "palm_gyro_y", "palm_gyro_z", "arm_gyro_x", "arm_gyro_y", "arm_gyro_z"
]
all_dimension_synths = [PalmAccDifSynth(), PalmGyroNormSynth(), ArmGyroNormSynth()]
all_window_sizes = [20, 30, 50, 70, 90]

# Multiprocessing requires this
if __name__ == "__main__":
    ctx = multiprocessing.get_context('spawn')

    # Initialize CSV for results
    csv_name = "compare_results.csv"
    csv_file = open(csv_name, mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    csv_writer.writerow(["ID", "Nosaukums", "Veziena tips", "Patiesi poz.", "Aplami poz.",
                         "Aplami neg.", "Precizitate", "Parklajums"])

    # Iniciet visus E2ERunner
    splits: list[str] = []
    splits_readable: list[str] = []
    split_sizes = [20, 500]
    split_folds = [5, 5]
    for idx, size in enumerate(split_sizes):
        for fold in range(split_folds[idx]):
            splits.append(f"split_final_{size}_F{fold}.pck")
            splits_readable.append(f"{size}-{fold}")

    runners: list[E2ERunner] = []

    for spl_idx, spl in enumerate(splits):
        for s in all_slices:
            name = f"Fragments {s.start}-{s.stop} (spl {splits_readable[spl_idx]})"
            runners.append(E2ERunner(name=name, dataset="fs_right_off",
                                     detector_builder_2=get_full_swing_rocket_with,
                                     db2_args={"crop": s, "split": spl}))
            runners.append(E2ERunner(name=name, dataset="put_right_off",
                                     detector_builder_2=get_putting_rocket_with,
                                     db2_args={"crop": s, "split": spl}))

        for dr in all_dimensions_to_remove:
            name = f"Dimensijas {dr[0]} (spl {splits_readable[spl_idx]})"
            runners.append(E2ERunner(name=name, dataset="fs_right_off",
                                     detector_builder_2=get_full_swing_rocket_with,
                                     db2_args={"dimensions_to_remove": dr[1], "split": spl}))
            runners.append(E2ERunner(name=name, dataset="put_right_off",
                                     detector_builder_2=get_putting_rocket_with,
                                     db2_args={"dimensions_to_remove": dr[1], "split": spl}))

        for ds in all_dimension_synths:
            name = f"Papild {ds.get_name()} (spl {splits_readable[spl_idx]})"
            runners.append(E2ERunner(name=name, dataset="fs_right_off",
                                     detector_builder_2=get_full_swing_rocket_with,
                                     db2_args={"dimensions_to_remove": all_dimensions,
                                               "synthesize_dimensions": [ds],
                                               "split": spl}))
            runners.append(E2ERunner(name=name, dataset="put_right_off",
                                     detector_builder_2=get_putting_rocket_with,
                                     db2_args={"dimensions_to_remove": all_dimensions,
                                               "synthesize_dimensions": [ds],
                                               "split": spl}))

        for ws in all_window_sizes:
            name = f"Loga izm. {ws} (spl {splits_readable[spl_idx]})"
            runners.append(E2ERunner(name=name, dataset="fs_right_off",
                                     detector_builder_2=get_full_swing_rocket_with,
                                     db2_args={"window_size": ws, "split": spl}))
            runners.append(E2ERunner(name=name, dataset="put_right_off",
                                     detector_builder_2=get_putting_rocket_with,
                                     db2_args={"window_size": ws, "split": spl}))

    # Iniciet multiprocessing
    pool = ctx.Pool(processes=4)

    # Palaist eksperimentus
    # for id, r in enumerate(runners):
    #
    #    save_results(id, r, results)
    processes = [pool.apply_async(execute_runner, args=(id, r,)) for id, r in enumerate(runners)]
    results = [p.get() for p in processes]

    for r in results:
        save_results(csv_writer, r[0], r[1], r[2])

    # Beigas aizvert failu
    csv_file.close()
