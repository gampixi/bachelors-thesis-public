import os
import sys
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

window: tkinter.Tk
canvas = None
fig: Figure
ax_accel_arm: Axes
ax_gyro_arm: Axes
ax_accel_palm: Axes
ax_gyro_palm: Axes

swing_paths: list[str] = []
current_swing_idx = 0
current_swing: list[WristSample] = None

mgresult: list[mg.MinigolfResult] = []


def _load_swing_list(dataset_kind: str):
    global swing_paths, current_swing_idx
    path_to = f"./dataset/{dataset_kind}/"
    pck_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.pck")]
    print(pck_files)
    for idx, pth in enumerate(pck_files):
        swing_paths.append(f"./dataset/{dataset_kind}/{pth}")
    current_swing_idx = 0


def _next_record():
    global swing_paths, current_swing_idx
    current_swing_idx += 1
    if current_swing_idx > len(swing_paths):
        current_swing_idx -= 1
    else:
        _load_current_record()


def _previous_record():
    global swing_paths, current_swing_idx
    current_swing_idx -= 1
    if current_swing_idx < 0:
        current_swing_idx += 1
    else:
        _load_current_record()


def _deactivate_record():
    global swing_paths, current_swing_idx
    new_name = f"{swing_paths[current_swing_idx]}.deactivated"
    print(f"Renaming {swing_paths[current_swing_idx]} to {new_name}")
    os.rename(swing_paths[current_swing_idx], new_name)


def _load_current_record():
    global swing_paths, current_swing_idx, current_swing
    if current_swing_idx >= len(swing_paths):
        print("No record for record index!")
        return

    filename = swing_paths[current_swing_idx].split("/")[-1]
    file_sv.set(filename)

    current_swing = swingDataInstance2wristSample(sdi_load(swing_paths[current_swing_idx]))
    _plot_current_snippet()
    _process_minigolf()


def _process_minigolf():
    global swing_paths, current_swing_idx, current_swing, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult
    if current_swing is None:
        print("No data for snippet index!")
        mgresult = []
        return

    mgresult = mg.run_full_configs(current_swing)
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
    #for idx, r in enumerate(mgresult):
    #    if r["result"] is not None:
    #        plot_mg_positions(r["result"], ax_accel_arm, plot_styles[idx][0], plot_styles[idx][1])
    #        plot_mg_positions(r["result"], ax_gyro_arm, plot_styles[idx][0], plot_styles[idx]#[1])
    #        plot_mg_positions(r["result"], ax_accel_palm, plot_styles[idx][0], plot_styles[idx][1])
    #        plot_mg_positions(r["result"], ax_gyro_palm, plot_styles[idx][0], plot_styles[idx][1])

    canvas.draw()


def _plot_current_snippet():
    global current_swing, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas
    ax_gyro_arm.clear()
    ax_accel_arm.clear()
    ax_gyro_palm.clear()
    ax_accel_palm.clear()

    if current_swing is None:
        print("No data for snippet index!")
        return

    plot_samples(current_swing, in_place_axes=np.array([[ax_accel_arm, ax_gyro_arm],[ax_accel_palm, ax_gyro_palm]]))

    canvas.draw()


def get_minigolf_matrix(runs: list[mg.MinigolfRun], dominantHand: DominantHand, wornHand: WornHand,
                        detector: mg.MinigolfDetector) -> str:
    if current_swing is None:
        print("No data for snippet index!")
        return ""
    for r in runs:
        if r["config"]["detector"].value == detector.value and r["config"][
            "dominantHand"].value == dominantHand.value and r["config"]["wornHand"].value == wornHand.value:
            return "YES" if r["result"] is not None else "NO"
    return "?"


def main():
    global current_swing, current_swing_idx, swing_paths, window, fig, ax_accel_arm, ax_gyro_arm, ax_accel_palm, ax_gyro_palm, canvas, mgresult
    global fs_right_off_sv, fs_left_off_sv, fs_right_dm_sv, fs_left_dm_sv, pt_right_off_sv, pt_left_off_sv, pt_right_dm_sv, pt_left_dm_sv
    global file_sv

    dataset_kind = sys.argv[1]

    window = tkinter.Tk()
    window.title(f'datu kopas pārskatītājs 9000 [{dataset_kind}]')
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

    quit_button = tkinter.Button(master=window, text="Quit", command=_quit)
    quit_button.pack(side=tkinter.BOTTOM)

    snippet_frame = tkinter.Frame(master=window)
    snippet_frame.pack(side=tkinter.BOTTOM)

    file_label = tkinter.Label(master=snippet_frame, textvariable=file_sv)
    file_label.pack(side=tkinter.LEFT)

    prev_file_button = tkinter.Button(master=snippet_frame, text="File Previous", command=_previous_record)
    prev_file_button.pack(side=tkinter.LEFT)

    next_file_button = tkinter.Button(master=snippet_frame, text="File Next", command=_next_record)
    next_file_button.pack(side=tkinter.LEFT)

    deactivate_file_button = tkinter.Button(master=snippet_frame, text="Deactivate file", command=_deactivate_record)
    deactivate_file_button.pack(side=tkinter.LEFT)

    detector_grid_frame = tkinter.Frame(master=window)
    detector_grid_frame.pack(side=tkinter.BOTTOM)

    fs_label = tkinter.Label(master=detector_grid_frame, text="FULL SW")
    pt_label = tkinter.Label(master=detector_grid_frame, text="PUTTING")
    off_label = tkinter.Label(master=detector_grid_frame, text="OFFHAND")
    dm_label = tkinter.Label(master=detector_grid_frame, text="DOMINANT")
    left_label = tkinter.Label(master=detector_grid_frame, text="LEFT")
    right_label = tkinter.Label(master=detector_grid_frame, text="RIGHT")

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

    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

    _load_swing_list(dataset_kind)
    _load_current_record()

    window.mainloop()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Provide a name of folder under ./dataset as command argument")
        exit(1)
    main()  # Magic speedup thanks to getting rid of global scope
