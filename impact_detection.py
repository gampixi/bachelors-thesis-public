import enum
from sensor_data_types import WristSample
import numpy as np


# https://stackoverflow.com/a/1751478/3482206
def chunks(l, n):
    n = max(1, n)
    return ((l[i:i+n], list(range(i, i+n))) for i in range(0, len(l), n))


# Find impacts and return the index of their position
def find_impacts(samples: list[WristSample],
                 palmVibrationThreshold: float = 1,
                 armGyroNormThreshold: float = 30,
                 minSpacing: int = 100) -> list[int]:
    # Peak value and index in impact
    impacts: list[(float, int)] = []
    # Check each window for peak impact
    for w, idxs in chunks(samples, minSpacing):
        w_acc_norms = []
        w_gyro_norms = []
        last_acc_norm = 0
        for i, s in enumerate(w):
            acc_norm = np.linalg.norm(s['palm_acc'])
            gyro_norm = np.linalg.norm(s['arm_gyro'])
            acc_norm_dif = abs(last_acc_norm - acc_norm)
            last_acc_norm = acc_norm
            w_acc_norms.append(acc_norm)
            w_gyro_norms.append(gyro_norm)

        current_peak_idx = -1
        current_peak_value = -1
        for i in range(len(w_acc_norms)):
            if (w_acc_norms[i] > palmVibrationThreshold) and (w_gyro_norms[i] > armGyroNormThreshold):
                if w_acc_norms[i] > current_peak_value:
                    current_peak_value = w_acc_norms[i]
                    current_peak_idx = i

        if current_peak_idx != -1:
            impacts.append((current_peak_value, current_peak_idx+idxs[0]))

    # Sort impacts by peak
    impacts.sort(key=lambda x: x[0], reverse=True)
    # Check if an impact is allowed to be detected on this sample
    '''
    next_impact_sample = 1
    for i, s in enumerate(samples):
        acc_norm = np.linalg.norm(s['palm_acc'])
        gyro_norm = np.linalg.norm(s['arm_gyro'])
        acc_norm_dif = abs(last_acc_norm - acc_norm)
        last_acc_norm = acc_norm
        if i < next_impact_sample:
            # 1st sample may report impact if extreme enough motion was at the beginning
            continue
        if (acc_norm_dif > palmVibrationThreshold) and (gyro_norm > armGyroNormThreshold):
            impacts.append(i)
            next_impact_sample = i + minSpacing
            print(f" * Found impact @ {i}, accNormDif: {acc_norm_dif:.2f} gyroNorm: {gyro_norm:.2f}")
    '''
    return list(map(lambda x: x[1], impacts))
    # return impacts


def impacts2snippets(samples: list[WristSample],
                     impacts: list[int],
                     before: int = 200,
                     after: int = 100) -> list[list[WristSample]]:
    snippets: list[list[WristSample]] = []
    for impact in impacts:
        min_range = impact - before
        max_range = impact + after
        if max_range > len(samples):
            max_range = len(samples)
            min_range = max_range - before - after
        if min_range < 0:
            min_range = 0
            max_range = before + after
        if max_range > len(samples):
            print(" ! Invalid - can not create snippet, will be skipped")
            continue
        snippets.append(samples[min_range:max_range])
    return snippets
