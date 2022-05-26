import os
import tkinter
from typing import TypedDict
import numpy as np
from matplotlib import pyplot as plt

import sensor_data_reader as sr
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

from e2e import get_samples, get_calibration
from e2e_detectors import E2ESwingMetadata
from sensor_data_types import CalibrationData, DominantHand, WornHand, WristSample
from impact_detection import *
import minigolf as mg
import glob
from swing_data_instance import *
from visualization import plot_samples


def plot_mg_positions(mgresult: mg.MinigolfResult | None, ax: Axes, color, style) -> list[Line2D]:
    if mgresult is None:
        return []

    a = ax.axvline(mgresult['address'], color=color, linestyle=style)
    t = ax.axvline(mgresult['top'], color=color, linestyle=style)
    i = ax.axvline(mgresult['impact'], color=color, linestyle=style)
    return [a, t, i]


fs_right_off_sv: tkinter.StringVar = None
fs_left_off_sv: tkinter.StringVar = None
fs_right_dm_sv: tkinter.StringVar = None
fs_left_dm_sv: tkinter.StringVar = None
pt_right_off_sv: tkinter.StringVar = None
pt_left_off_sv: tkinter.StringVar = None
pt_right_dm_sv: tkinter.StringVar = None
pt_left_dm_sv: tkinter.StringVar = None
file_sv: tkinter.StringVar = None
pos_entry: tkinter.Entry = None

window: tkinter.Tk
canvas = None
fig: Figure
ax_accel_arm: Axes
ax_gyro_arm: Axes
ax_accel_palm: Axes
ax_gyro_palm: Axes


class RawDataRecord(TypedDict):
    data_path: str
    calibration_path: str


records: list[RawDataRecord] = []
current_samples: list[WristSample] = []
current_record_idx = 0


def _load_unprocessed_list():
    global records, current_record_idx
    path_to = "./e2e_raw_tomark/"
    bin_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.bin")]
    cal_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.cal")]
    print(bin_files)
    print(cal_files)
    for bix, b in enumerate(bin_files):
        common_part = b[0:-4]
        matching_cal_name = f"{common_part}_CD.cal"
        for idx, c in enumerate(cal_files):
            if c == matching_cal_name:
                records.append({
                    "data_path": f"{path_to}{b}",
                    "calibration_path": f"{path_to}{c}",
                })
                bin_files[bix] = "USED UP NAME"
                cal_files[idx] = "USED UP NAME"
                break
    current_record_idx = 0


mgresult: list[mg.MinigolfResult] = []


def _next_record():
    global records, current_record_idx
    current_record_idx += 1
    if current_record_idx > len(records):
        current_record_idx -= 1
    else:
        _load_current_record()


def _previous_record():
    global records, current_record_idx
    current_record_idx -= 1
    if current_record_idx < 0:
        current_record_idx += 1
    else:
        _load_current_record()


def _load_current_record():
    if current_record_idx >= len(records):
        print("No record for record index!")
        return

    filename = records[current_record_idx]["data_path"].split("/")[-1]
    file_sv.set(filename)

    global current_samples
    samples = get_samples(records[current_record_idx]["data_path"])
    calibration = get_calibration(records[current_record_idx]["calibration_path"])
    sr.apply_calibration(samples, calibration)
    current_samples = samples
    _plot_current_samples()
    _process_minigolf()


def _process_minigolf():
    global current_samples, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult

    mgresult = mg.run_full_configs(current_samples)
    print(f" * Minigolf result: {mgresult}")
    fs_right_off_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.RIGHT, WornHand.OFFHAND, mg.MinigolfDetector.FULLSWING))
    fs_left_off_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.LEFT, WornHand.OFFHAND, mg.MinigolfDetector.FULLSWING))
    fs_right_dm_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.RIGHT, WornHand.DOMINANT, mg.MinigolfDetector.FULLSWING))
    fs_left_dm_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.LEFT, WornHand.DOMINANT, mg.MinigolfDetector.FULLSWING))
    pt_right_off_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.RIGHT, WornHand.OFFHAND, mg.MinigolfDetector.PUTTING))
    pt_left_off_sv.set(get_minigolf_matrix(mgresult, DominantHand.LEFT, WornHand.OFFHAND, mg.MinigolfDetector.PUTTING))
    pt_right_dm_sv.set(
        get_minigolf_matrix(mgresult, DominantHand.RIGHT, WornHand.DOMINANT, mg.MinigolfDetector.PUTTING))
    pt_left_dm_sv.set(get_minigolf_matrix(mgresult, DominantHand.LEFT, WornHand.DOMINANT, mg.MinigolfDetector.PUTTING))

    pos_entry.delete(0, tkinter.END)
    pos_entry.insert(0, "")
    plot_styles = [('c', '-',), ('m', '-',), ('y', '-',), ('k', '-',), ('c', '--',), ('m', '--',), ('y', '--',),
                   ('k', '--',)]
    for idx, r in enumerate(mgresult):
        if r["result"] is not None:
            imp = r["result"]["impact"]
            pos_entry.insert(0, f"{imp},")
            plot_mg_positions(r["result"], ax_accel_arm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_gyro_arm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_accel_palm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_gyro_palm, plot_styles[idx][0], plot_styles[idx][1])

    canvas.draw()


def _plot_current_samples():
    global current_samples, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas
    ax_gyro_arm.clear()
    ax_accel_arm.clear()
    ax_gyro_palm.clear()
    ax_accel_palm.clear()

    plot_samples(current_samples, in_place_axes=np.array([[ax_accel_arm, ax_gyro_arm],[ax_accel_palm, ax_gyro_palm]]))

    canvas.draw()


def _save_current_sample(isNot: bool, swingType: SwingType | None,
                         dominantHand: DominantHand | None,
                         wornHand: WornHand | None):
    print("SAVE")
    global records, current_record_idx
    folder_name = ""
    if isNot:
        folder_name = "not"
    else:
        if swingType == SwingType.FULL_SWING:
            folder_name += "fs_"
        else:
            folder_name += "put_"
        if dominantHand == DominantHand.RIGHT:
            folder_name += "right_"
        else:
            folder_name += "left_"
        if wornHand == WornHand.OFFHAND:
            folder_name += "off"
        else:
            folder_name += "dm"

    new_data_name = records[current_record_idx]['data_path'].replace('e2e_raw_tomark', f'e2e_dataset/{folder_name}')
    new_cal_name = records[current_record_idx]['calibration_path'].replace('e2e_raw_tomark',
                                                                           f'e2e_dataset/{folder_name}')
    new_meta_name = new_data_name.replace(".bin", ".pck")
    print(f"Moving to {new_data_name} and {new_cal_name}")
    os.rename(records[current_record_idx]['data_path'], new_data_name)
    os.rename(records[current_record_idx]['calibration_path'], new_cal_name)
    if not isNot:
        # Create file containing swing positions
        pos_strs = pos_entry.get().split(",")
        pos = []
        for p in pos_strs:
            try:
                pos.append(int(p))
            except:
                continue
        print(pos)
        meta = E2ESwingMetadata(impact_positions=pos)
        pickled = pickle.dumps(meta, fix_imports=False)
        with open(new_meta_name, "wb") as file:
            file.write(pickled)

    records[current_record_idx]['data_path'] = new_data_name
    records[current_record_idx]['calibration_path'] = new_cal_name


def _save_fs_right_off():
    _save_current_sample(False, SwingType.FULL_SWING, DominantHand.RIGHT, WornHand.OFFHAND)


def _save_fs_right_dm():
    _save_current_sample(False, SwingType.FULL_SWING, DominantHand.RIGHT, WornHand.DOMINANT)


def _save_fs_left_off():
    _save_current_sample(False, SwingType.FULL_SWING, DominantHand.LEFT, WornHand.OFFHAND)


def _save_fs_left_dm():
    _save_current_sample(False, SwingType.FULL_SWING, DominantHand.LEFT, WornHand.DOMINANT)


def _save_put_right_off():
    _save_current_sample(False, SwingType.PUTTING, DominantHand.RIGHT, WornHand.OFFHAND)


def _save_put_right_dm():
    _save_current_sample(False, SwingType.PUTTING, DominantHand.RIGHT, WornHand.DOMINANT)


def _save_put_left_off():
    _save_current_sample(False, SwingType.PUTTING, DominantHand.LEFT, WornHand.OFFHAND)


def _save_put_left_dm():
    _save_current_sample(False, SwingType.PUTTING, DominantHand.LEFT, WornHand.DOMINANT)


def _save_not_swing():
    _save_current_sample(True, None, None, None)


def get_minigolf_matrix(runs: list[mg.MinigolfRun], dominantHand: DominantHand, wornHand: WornHand,
                        detector: mg.MinigolfDetector) -> str:
    for r in runs:
        if r["config"]["detector"].value == detector.value and r["config"][
            "dominantHand"].value == dominantHand.value and r["config"]["wornHand"].value == wornHand.value:
            return "+++" if r["result"] is not None else "-"
    return "?"


def main():
    global current_samples, window, fig, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult
    global fs_right_off_sv, fs_left_off_sv, fs_right_dm_sv, fs_left_dm_sv, pt_right_off_sv, pt_left_off_sv, pt_right_dm_sv, pt_left_dm_sv
    global file_sv, pos_entry

    window = tkinter.Tk()
    window.title('Validācijas datu kopas būvētājs 9001')
    window.geometry('900x900')

    fs_right_off_sv = tkinter.StringVar()
    fs_left_off_sv = tkinter.StringVar()
    fs_right_dm_sv = tkinter.StringVar()
    fs_left_dm_sv = tkinter.StringVar()
    pt_right_off_sv = tkinter.StringVar()
    pt_left_off_sv = tkinter.StringVar()
    pt_right_dm_sv = tkinter.StringVar()
    pt_left_dm_sv = tkinter.StringVar()
    file_sv = tkinter.StringVar()

    fig, ax = plt.subplots(2, 2, sharey='row')
    ax_accel_arm = ax[0, 0]
    ax_gyro_arm = ax[0, 1]
    ax_accel_palm = ax[1, 0]
    ax_gyro_palm = ax[1, 1]

    canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    def _quit():
        window.quit()
        window.destroy()

    quit_button = tkinter.Button(master=window, text="Aizvērt", command=_quit)
    quit_button.pack(side=tkinter.BOTTOM)

    snippet_frame = tkinter.Frame(master=window)
    snippet_frame.pack(side=tkinter.BOTTOM)

    file_label = tkinter.Label(master=snippet_frame, textvariable=file_sv)
    file_label.pack(side=tkinter.LEFT)

    prev_file_button = tkinter.Button(master=snippet_frame, text="Iepr. fails", command=_previous_record)
    prev_file_button.pack(side=tkinter.LEFT)

    next_file_button = tkinter.Button(master=snippet_frame, text="Nāk. fails", command=_next_record)
    next_file_button.pack(side=tkinter.LEFT)

    pos_entry = tkinter.Entry(master=snippet_frame)
    pos_entry.pack(side=tkinter.LEFT)

    detector_grid_frame = tkinter.Frame(master=window)
    detector_grid_frame.pack(side=tkinter.BOTTOM)

    fs_label = tkinter.Label(master=detector_grid_frame, text="Pilns v.")
    pt_label = tkinter.Label(master=detector_grid_frame, text="Ripin.")
    off_label = tkinter.Label(master=detector_grid_frame, text="Nedomin.")
    dm_label = tkinter.Label(master=detector_grid_frame, text="Domin.")
    left_label = tkinter.Label(master=detector_grid_frame, text="Kreiļu")
    right_label = tkinter.Label(master=detector_grid_frame, text="Labrč.")

    fs_label.grid(row=0, column=1, columnspan=2)
    off_label.grid(row=1, column=1)
    dm_label.grid(row=1, column=2)
    right_label.grid(row=2, column=0)
    left_label.grid(row=3, column=0)

    pt_label.grid(row=0, column=3, columnspan=2)

    fs_right_off = tkinter.Label(master=detector_grid_frame, textvariable=fs_right_off_sv)
    fs_right_off.grid(row=2, column=1)

    fs_left_off = tkinter.Label(master=detector_grid_frame, textvariable=fs_left_off_sv)
    fs_left_off.grid(row=3, column=1)

    fs_right_dm = tkinter.Label(master=detector_grid_frame, textvariable=fs_right_dm_sv)
    fs_right_dm.grid(row=2, column=2)

    fs_left_dm = tkinter.Label(master=detector_grid_frame, textvariable=fs_left_dm_sv)
    fs_left_dm.grid(row=3, column=2)

    pt_right_off = tkinter.Label(master=detector_grid_frame, textvariable=pt_right_off_sv)
    pt_right_off.grid(row=2, column=3)

    pt_left_off = tkinter.Label(master=detector_grid_frame, textvariable=pt_left_off_sv)
    pt_left_off.grid(row=3, column=3)

    pt_right_dm = tkinter.Label(master=detector_grid_frame, textvariable=pt_right_dm_sv)
    pt_right_dm.grid(row=2, column=4)

    pt_left_dm = tkinter.Label(master=detector_grid_frame, textvariable=pt_left_dm_sv)
    pt_left_dm.grid(row=3, column=4)

    save_as_fs_right_off_btn = tkinter.Button(master=detector_grid_frame, text="PV / LAB / NED",
                                              command=_save_fs_right_off)
    save_as_fs_left_off_btn = tkinter.Button(master=detector_grid_frame, text="PV / KR / NED",
                                             command=_save_fs_left_off)
    save_as_fs_right_dm_btn = tkinter.Button(master=detector_grid_frame, text="PV / LAB / DOMIN.",
                                             command=_save_fs_right_dm)
    save_as_fs_left_dm_btn = tkinter.Button(master=detector_grid_frame, text="PV / KR / DOMIN.",
                                            command=_save_fs_left_dm)
    save_as_put_right_off_btn = tkinter.Button(master=detector_grid_frame, text="RIP / LAB / NED",
                                               command=_save_put_right_off)
    save_as_put_left_off_btn = tkinter.Button(master=detector_grid_frame, text="RIP / KR / NED",
                                              command=_save_put_left_off)
    save_as_put_right_dm_btn = tkinter.Button(master=detector_grid_frame, text="RIP / LAB / DOMIN.",
                                              command=_save_put_right_dm)
    save_as_put_left_dm_btn = tkinter.Button(master=detector_grid_frame, text="RIP / KR / DOMIN.",
                                             command=_save_put_left_dm)
    save_as_not_swing_btn = tkinter.Button(master=detector_grid_frame, text="NESITIENS",
                                           command=_save_not_swing)

    save_as_fs_right_off_btn.grid(row=2, column=5)
    save_as_fs_left_off_btn.grid(row=3, column=5)
    save_as_fs_right_dm_btn.grid(row=2, column=6)
    save_as_fs_left_dm_btn.grid(row=3, column=6)
    save_as_put_right_off_btn.grid(row=2, column=7)
    save_as_put_left_off_btn.grid(row=3, column=7)
    save_as_put_right_dm_btn.grid(row=2, column=8)
    save_as_put_left_dm_btn.grid(row=3, column=8)
    save_as_not_swing_btn.grid(row=1, column=5, columnspan=4)

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    _load_unprocessed_list()
    _load_current_record()

    window.mainloop()


if __name__ == "__main__":
    main()  # Magic speedup thanks to getting rid of global scope
