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
snippet_sv: tkinter.StringVar = None
file_sv: tkinter.StringVar = None

window: tkinter.Tk
canvas = None
fig: Figure
ax_accel_arm: Axes
ax_gyro_arm: Axes
ax_accel_palm: Axes
ax_gyro_palm: Axes

current_snippets: list[list[WristSample]] = []
current_snippets_save_state: list[str] = []
current_snippet_idx = 0


class RawDataRecord(TypedDict):
    data_path: str
    calibration_path: str


records: list[RawDataRecord] = []
current_record_idx = 0


def _load_unprocessed_list():
    global records, current_record_idx
    path_to = "./unprocessed_raw/"
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


def _next_and_move_record():
    global records, current_record_idx
    new_data_name = records[current_record_idx]['data_path'].replace('unprocessed_raw', 'processed_raw')
    new_cal_name = records[current_record_idx]['calibration_path'].replace('unprocessed_raw', 'processed_raw')
    print(f"Moving to {new_data_name} and {new_cal_name}")
    os.rename(records[current_record_idx]['data_path'], new_data_name)
    os.rename(records[current_record_idx]['calibration_path'], new_cal_name)
    records.pop(0)
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

    global current_snippets, current_snippet_idx, current_snippets_save_state
    samples = get_samples(records[current_record_idx]["data_path"])
    calibration = get_calibration(records[current_record_idx]["calibration_path"])
    sr.apply_calibration(samples, calibration)
    current_snippets = impacts2snippets(samples, find_impacts(samples))
    if len(current_snippets) == 0:
        current_snippets = [samples]
        current_snippets_save_state = []
    else:
        current_snippets_save_state = [None for x in current_snippets]
    current_snippet_idx = 0
    _previous_snippet()


def _set_snippet_label():
    snippet_sv.set(f"{current_snippet_idx + 1}/{len(current_snippets)}")


def _next_snippet():
    global current_snippets, current_snippet_idx
    current_snippet_idx += 1
    if current_snippet_idx >= len(current_snippets):
        current_snippet_idx -= 1
    print(f"Current snippet idx: {current_snippet_idx}")
    _plot_current_snippet()
    _process_minigolf()
    _set_snippet_label()


def _previous_snippet():
    global current_snippets, current_snippet_idx
    current_snippet_idx -= 1
    if current_snippet_idx < 0:
        current_snippet_idx = 0
    print(f"Current snippet idx: {current_snippet_idx}")
    _plot_current_snippet()
    _process_minigolf()
    _set_snippet_label()


def _process_minigolf():
    global current_snippets, current_snippet_idx, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult
    if current_snippet_idx >= len(current_snippets):
        print("No data for snippet index!")
        mgresult = []
        return

    mgresult = mg.run_full_configs(current_snippets[current_snippet_idx])
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

    plot_styles = [('c', '-',), ('m', '-',), ('y', '-',), ('k', '-',), ('c', '--',), ('m', '--',), ('y', '--',),
                   ('k', '--',)]
    for idx, r in enumerate(mgresult):
        if r["result"] is not None:
            plot_mg_positions(r["result"], ax_accel_arm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_gyro_arm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_accel_palm, plot_styles[idx][0], plot_styles[idx][1])
            plot_mg_positions(r["result"], ax_gyro_palm, plot_styles[idx][0], plot_styles[idx][1])

    canvas.draw()


def _plot_current_snippet():
    global current_snippets, current_snippet_idx, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas
    ax_gyro_arm.clear()
    ax_accel_arm.clear()
    ax_gyro_palm.clear()
    ax_accel_palm.clear()

    if current_snippet_idx >= len(current_snippets):
        print("No data for snippet index!")
        return

    plot_samples(current_snippets[current_snippet_idx],
                 in_place_axes=np.array([[ax_accel_arm, ax_gyro_arm], [ax_accel_palm, ax_gyro_palm]]))

    canvas.draw()


def _save_current_snippet(swingType: SwingType | None,
                          dominantHand: DominantHand | None,
                          wornHand: WornHand | None):
    if current_snippet_idx >= len(current_snippets):
        print("No data for snippet index!")
        return
    if current_snippets_save_state[current_snippet_idx] is not None:
        _unsave_current_snippet()

    sdi = wristSample2swingDataInstance(current_snippets[current_snippet_idx],
                                        swingType,
                                        dominantHand,
                                        wornHand)
    filename = sdi_save(sdi)
    current_snippets_save_state[current_snippet_idx] = filename


def _unsave_current_snippet():
    sp = current_snippets_save_state[current_snippet_idx]
    print(f"Deleting {sp}...")
    current_snippets_save_state[current_snippet_idx] = None
    os.remove(sp)


def _save_fs_right_off():
    _save_current_snippet(SwingType.FULL_SWING, DominantHand.RIGHT, WornHand.OFFHAND)


def _save_fs_right_dm():
    _save_current_snippet(SwingType.FULL_SWING, DominantHand.RIGHT, WornHand.DOMINANT)


def _save_fs_left_off():
    _save_current_snippet(SwingType.FULL_SWING, DominantHand.LEFT, WornHand.OFFHAND)


def _save_fs_left_dm():
    _save_current_snippet(SwingType.FULL_SWING, DominantHand.LEFT, WornHand.DOMINANT)


def _save_put_right_off():
    _save_current_snippet(SwingType.PUTTING, DominantHand.RIGHT, WornHand.OFFHAND)


def _save_put_right_dm():
    _save_current_snippet(SwingType.PUTTING, DominantHand.RIGHT, WornHand.DOMINANT)


def _save_put_left_off():
    _save_current_snippet(SwingType.PUTTING, DominantHand.LEFT, WornHand.OFFHAND)


def _save_put_left_dm():
    _save_current_snippet(SwingType.PUTTING, DominantHand.LEFT, WornHand.DOMINANT)


def _save_not_swing():
    _save_current_snippet(None, None, None)


def get_minigolf_matrix(runs: list[mg.MinigolfRun], dominantHand: DominantHand, wornHand: WornHand,
                        detector: mg.MinigolfDetector) -> str:
    if current_snippet_idx >= len(current_snippets):
        print("No data for snippet index!")
        return ""
    for r in runs:
        if r["config"]["detector"].value == detector.value and r["config"][
            "dominantHand"].value == dominantHand.value and r["config"]["wornHand"].value == wornHand.value:
            return "+++" if r["result"] is not None else "-"
    return "?"


def main():
    global current_snippets, current_snippet_idx, window, fig, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult
    global fs_right_off_sv, fs_left_off_sv, fs_right_dm_sv, fs_left_dm_sv, pt_right_off_sv, pt_left_off_sv, pt_right_dm_sv, pt_left_dm_sv
    global snippet_sv, file_sv

    window = tkinter.Tk()
    window.title('Apmācības kopas būvētājs 3000')
    window.geometry('900x900')

    fs_right_off_sv = tkinter.StringVar()
    fs_left_off_sv = tkinter.StringVar()
    fs_right_dm_sv = tkinter.StringVar()
    fs_left_dm_sv = tkinter.StringVar()
    pt_right_off_sv = tkinter.StringVar()
    pt_left_off_sv = tkinter.StringVar()
    pt_right_dm_sv = tkinter.StringVar()
    pt_left_dm_sv = tkinter.StringVar()
    snippet_sv = tkinter.StringVar()
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

    snippet_label = tkinter.Label(master=snippet_frame, textvariable=snippet_sv)
    snippet_label.pack(side=tkinter.LEFT)

    file_label = tkinter.Label(master=snippet_frame, textvariable=file_sv)
    file_label.pack(side=tkinter.LEFT)

    prev_snip_button = tkinter.Button(master=snippet_frame, text="Iepr. fragments", command=_previous_snippet)
    prev_snip_button.pack(side=tkinter.LEFT)

    next_snip_button = tkinter.Button(master=snippet_frame, text="Nākamais fragments", command=_next_snippet)
    next_snip_button.pack(side=tkinter.LEFT)

    prev_file_button = tkinter.Button(master=snippet_frame, text="Iepr. fails", command=_previous_record)
    prev_file_button.pack(side=tkinter.LEFT)

    next_and_move_file_button = tkinter.Button(master=snippet_frame, text="Nāk. fails & pārcelt",
                                               command=_next_and_move_record)
    next_and_move_file_button.pack(side=tkinter.LEFT)

    next_file_button = tkinter.Button(master=snippet_frame, text="Nāk. fails", command=_next_record)
    next_file_button.pack(side=tkinter.LEFT)

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
    save_revert_btn = tkinter.Button(master=detector_grid_frame, text="ATSAGLABĀT",
                                     command=_unsave_current_snippet)

    save_as_fs_right_off_btn.grid(row=2, column=5)
    save_as_fs_left_off_btn.grid(row=3, column=5)
    save_as_fs_right_dm_btn.grid(row=2, column=6)
    save_as_fs_left_dm_btn.grid(row=3, column=6)
    save_as_put_right_off_btn.grid(row=2, column=7)
    save_as_put_left_off_btn.grid(row=3, column=7)
    save_as_put_right_dm_btn.grid(row=2, column=8)
    save_as_put_left_dm_btn.grid(row=3, column=8)
    save_as_not_swing_btn.grid(row=1, column=5, columnspan=4)
    save_revert_btn.grid(row=0, column=5, columnspan=4)

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    _load_unprocessed_list()
    _load_current_record()

    window.mainloop()


if __name__ == "__main__":
    main()  # Magic speedup thanks to getting rid of global scope
