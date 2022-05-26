from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sensor_data_types import WristSample


def num_to_text_label(num_label):
    if num_label == 0:
        return "Not a swing"
    nl = num_label - 1
    dh = "Right" if (nl & 1) != 0 else "Left"
    wh = "Off." if (nl & 2) != 0 else "Dom."
    st = "Pt." if (nl & 4) != 0 else "Fs."
    return f"{st} {wh} {dh}"


def plot_cm(test_classes, test_predictions, class_labels):
    text_labels = list(map(lambda x: num_to_text_label(x), class_labels))
    cm = confusion_matrix(test_classes, test_predictions, labels=class_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=text_labels)
    disp.plot(cmap=plt.cm.Blues)


_x_ls = "-"
_y_ls = "--"
_z_ls = ":"
_lthicc = 1.5


def plot_arm_acceleration(samples: list[WristSample], ax: Axes) -> list[Line2D]:
    xx = [x for x in map(lambda x: x['arm_acc'][0], samples)]
    yy = [x for x in map(lambda x: x['arm_acc'][1], samples)]
    zz = [x for x in map(lambda x: x['arm_acc'][2], samples)]
    x_line = ax.plot(xx, 'r', label="Paātrinājums X", linestyle=_x_ls, linewidth=_lthicc)
    y_line = ax.plot(yy, 'g', label="Paātrinājums Y", linestyle=_y_ls, linewidth=_lthicc)
    z_line = ax.plot(zz, 'b', label="Paātrinājums Z", linestyle=_z_ls, linewidth=_lthicc)
    return [x_line[0], y_line[0], z_line[0]]


def plot_arm_rotation(samples: list[WristSample], ax: Axes) -> list[Line2D]:
    xx = [x for x in map(lambda x: x['arm_gyro'][0], samples)]
    yy = [x for x in map(lambda x: x['arm_gyro'][1], samples)]
    zz = [x for x in map(lambda x: x['arm_gyro'][2], samples)]
    x_line = ax.plot(xx, 'r', label="Leņķ. ātrums X", linestyle=_x_ls, linewidth=_lthicc)
    y_line = ax.plot(yy, 'g', label="Leņķ. ātrums Y", linestyle=_y_ls, linewidth=_lthicc)
    z_line = ax.plot(zz, 'b', label="Leņķ. ātrums Z", linestyle=_z_ls, linewidth=_lthicc)
    return [x_line[0], y_line[0], z_line[0]]


def plot_palm_acceleration(samples: list[WristSample], ax: Axes) -> list[Line2D]:
    xx = [x for x in map(lambda x: x['palm_acc'][0], samples)]
    yy = [x for x in map(lambda x: x['palm_acc'][1], samples)]
    zz = [x for x in map(lambda x: x['palm_acc'][2], samples)]
    x_line = ax.plot(xx, 'r', label="Paātrinājums X", linestyle=_x_ls, linewidth=_lthicc)
    y_line = ax.plot(yy, 'g', label="Paātrinājums Y", linestyle=_y_ls, linewidth=_lthicc)
    z_line = ax.plot(zz, 'b', label="Paātrinājums Z", linestyle=_z_ls, linewidth=_lthicc)
    return [x_line[0], y_line[0], z_line[0]]


def plot_palm_rotation(samples: list[WristSample], ax: Axes) -> list[Line2D]:
    xx = [x for x in map(lambda x: x['palm_gyro'][0], samples)]
    yy = [x for x in map(lambda x: x['palm_gyro'][1], samples)]
    zz = [x for x in map(lambda x: x['palm_gyro'][2], samples)]
    x_line = ax.plot(xx, 'r', label="Leņķ. ātrums X", linestyle=_x_ls, linewidth=_lthicc)
    y_line = ax.plot(yy, 'g', label="Leņķ. ātrums Y", linestyle=_y_ls, linewidth=_lthicc)
    z_line = ax.plot(zz, 'b', label="Leņķ. ātrums Z", linestyle=_z_ls, linewidth=_lthicc)
    return [x_line[0], y_line[0], z_line[0]]


def plot_markers(markers: list[int], ax: Axes, color, style) -> list[Line2D]:
    ret = []
    for m in markers:
        ret.append(ax.axvline(m, color=color, linestyle=style))
    return ret


def plot_samples(samples: list[WristSample],
                 det_markers: list[int] = [],
                 truth_markers: list[int] = [],
                 legend: bool = True,
                 grid: bool = True,
                 in_place_axes: Any | None = None):
    ax = None
    fig = None
    if in_place_axes is None:
        fig, ax = plt.subplots(2, 2, sharey='row')
        fig.set_size_inches(14, 8)
    else:
        ax = in_place_axes
    plot_arm_acceleration(samples, ax[0, 0])
    plot_arm_rotation(samples, ax[1, 0])
    plot_palm_acceleration(samples, ax[0, 1])
    plot_palm_rotation(samples, ax[1, 1])

    if grid:
        ax[0, 0].grid()
        ax[1, 0].grid()
        ax[0, 1].grid()
        ax[1, 1].grid()

    if legend:
        ax[0, 0].legend()
        ax[0, 0].set_xlabel("Mērījums")
        ax[0, 0].set_ylabel("Rokas paātrinājums (m/s^2)")

        ax[1, 0].legend()
        ax[1, 0].set_xlabel("Mērījums")
        ax[1, 0].set_ylabel("Rokas leņķiskais ātrums (deg/s)")

        ax[0, 1].legend()
        ax[0, 1].set_xlabel("Mērījums")
        ax[0, 1].set_ylabel("Plaukstas paātrinājums (m/s^2)")

        ax[1, 1].legend()
        ax[1, 1].set_xlabel("Mērījums")
        ax[1, 1].set_ylabel("Plaukstas leņķiskais ātrums (deg/s)")

    det_style = "-"
    truth_style = "--"
    det_color = "k"
    truth_color = "m"
    plot_markers(det_markers, ax[0, 0], det_color, det_style)
    plot_markers(det_markers, ax[1, 0], det_color, det_style)
    plot_markers(det_markers, ax[0, 1], det_color, det_style)
    plot_markers(det_markers, ax[1, 1], det_color, det_style)
    plot_markers(truth_markers, ax[0, 0], truth_color, truth_style)
    plot_markers(truth_markers, ax[1, 0], truth_color, truth_style)
    plot_markers(truth_markers, ax[0, 1], truth_color, truth_style)
    plot_markers(truth_markers, ax[1, 1], truth_color, truth_style)

    if in_place_axes is not None:
        return None
    return fig, ax
