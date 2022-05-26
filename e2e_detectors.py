import subprocess
from launchpad import *
from sensor_data_types import *
from minigolf import MinigolfConfig, MinigolfResult, MinigolfDetector
from collections import deque
import numpy as np


class E2ESwingMetadata(TypedDict):
    impact_positions: list[int]


_default_split = "split_final_500_F0.pck"


class E2EDetector:
    def get_name(self) -> str:
        raise "Detector doesn't override get_name"

    # Returns None if no swing detected. Otherwise, returns how far back impact occurred.
    def add_sample(self, sample: WristSample) -> int | None:
        pass


def geq_sign_invariant(a, b):
    if b < 0:
        return a <= b
    else:
        return a >= b


def detect_threshold(impact_window: list[WristSample],
                     palm_vibration_threshold: float = 2,
                     arm_gyro_x_threshold: float | None = 40,
                     palm_gyro_z_dif_threshold: float | None = None) -> int | None:
    w_acc_dif_norms = []
    w_gyro_x_vals = []
    w_gyro_z_vals = []
    w_gyro_z_difs = []
    last_acc = None
    last_gyro_z = None
    for i, s in enumerate(impact_window):
        acc = s['palm_acc']

        if last_acc is None:
            last_acc = acc
            last_gyro_z = s['palm_gyro'][2]

        acc_dif_norm = np.linalg.norm([acc[0] - last_acc[0], acc[1] - last_acc[1], acc[2] - last_acc[2]])
        gyro_z_dif = s['palm_gyro'][2] - last_gyro_z
        gyro_x = s['arm_gyro'][0]
        gyro_z = s['palm_gyro'][2]
        last_acc = s['palm_acc']
        last_gyro_z = gyro_z
        w_acc_dif_norms.append(acc_dif_norm)
        w_gyro_x_vals.append(gyro_x)
        w_gyro_z_vals.append(gyro_z)
        w_gyro_z_difs.append(gyro_z_dif)

    current_peak_idx = None
    current_peak_value = -9999999
    if palm_gyro_z_dif_threshold is None:
        # Use basic impact detection based on acceleration peak
        for i in range(len(w_acc_dif_norms)):
            if (w_acc_dif_norms[i] >= palm_vibration_threshold) \
                    and (True if arm_gyro_x_threshold is None else w_gyro_x_vals[i] >= arm_gyro_x_threshold):
                if w_acc_dif_norms[i] > current_peak_value:
                    current_peak_value = w_acc_dif_norms[i]
                    current_peak_idx = i
    else:
        # Use impact detection based on combined score
        for i in range(len(w_acc_dif_norms)):
            if (w_acc_dif_norms[i] >= palm_vibration_threshold) \
                    and (True if arm_gyro_x_threshold is None else w_gyro_x_vals[i] >= arm_gyro_x_threshold)\
                    and geq_sign_invariant(w_gyro_z_difs[i], palm_gyro_z_dif_threshold):
                # print(f"Ac dif: {w_acc_dif_norms[i]}, gZ dif: {w_gyro_z_difs[i]}")
                acc_weight = 20
                gyro_x_weight = 0
                gyro_z_weight = -1
                score = w_acc_dif_norms[i] * acc_weight + w_gyro_x_vals[i] * gyro_x_weight + w_gyro_z_difs[
                    i] * gyro_z_weight
                if score > current_peak_value:
                    # print(f"New peak score: {score} @ {i}")
                    # print(f"+ Acc dif\t: {w_acc_dif_norms[i] * acc_weight}")
                    # print(f"+ Gyro X\t: {w_gyro_x_vals[i] * gyro_x_weight}")
                    # print(f"+ gZ dif\t: {w_gyro_z_difs[i] * gyro_z_weight}")
                    current_peak_value = score
                    current_peak_idx = i

    if current_peak_idx is None:
        return None
    # print(f"Window evaluated with peak @ {current_peak_idx}")
    return len(impact_window) - current_peak_idx


class E2EThreshold(E2EDetector):
    def __init__(self,
                 name: str = "Threshold",
                 window_size: int = 50,
                 cooldown_period: int = 75,
                 palm_vibration_threshold: float = 2,
                 arm_gyro_x_threshold: float | None = 40,
                 palm_gyro_z_dif_threshold: float | None = None):
        self.name = name
        self.buffer = deque([])
        self.sample_count = 0
        self.window_size = window_size
        self.window_period = (self.window_size // 4) * 3
        self.next_window = self.window_period
        self.cooldown_period = cooldown_period
        self.cooldown_timer = 0
        # These should be larger than for "impact pre-detector"
        self.palm_vibration_threshold = palm_vibration_threshold
        self.arm_gyro_x_threshold = arm_gyro_x_threshold
        self.palm_gyro_z_dif_threshold = palm_gyro_z_dif_threshold

    def get_name(self) -> str:
        return self.name

    def iterate_impact(self) -> int | None:
        # Returns if this window contained an impact and if so, how many samples back was it
        self.next_window -= 1
        self.sample_count += 1
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return None
        if self.next_window <= 0:
            self.next_window = self.window_period
            if len(self.buffer) < self.window_size:
                return None
            else:
                return detect_threshold(
                    list(self.buffer)[(len(self.buffer) - self.window_size):],
                    palm_vibration_threshold=self.palm_vibration_threshold,
                    arm_gyro_x_threshold=self.arm_gyro_x_threshold,
                    palm_gyro_z_dif_threshold=self.palm_gyro_z_dif_threshold
                )
        else:
            return None

    def add_sample(self, sample: WristSample) -> int | None:
        self.buffer.append(sample)
        if len(self.buffer) > (self.window_size * 2):
            self.buffer.popleft()
        imp_result = self.iterate_impact()
        if imp_result is not None:
            self.cooldown_timer = self.cooldown_period
            return self.sample_count - imp_result
        else:
            return None


class E2EMinigolf(E2EDetector):
    def __init__(self,
                 dominantHand: DominantHand,
                 wornHand: WornHand,
                 detector: MinigolfDetector,
                 name: str = "Minigolf"):
        self.name = name
        self.sample_count = 0
        self.mgp = subprocess.Popen("./minigolf",
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    bufsize=1)

        config_line = f"{dominantHand.value} {wornHand.value} {detector.value}\n"
        self.mgp.stdin.write(config_line)

    def __del__(self):
        self.mgp.terminate()

    def get_name(self) -> str:
        return self.name

    def add_sample(self, sample: WristSample) -> int | None:
        sample_line = "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(
            sample["arm_acc"][0], sample["arm_acc"][1], sample["arm_acc"][2],
            sample["arm_gyro"][0], sample["arm_gyro"][1], sample["arm_gyro"][2],
            sample["palm_acc"][0], sample["palm_acc"][1], sample["palm_acc"][2],
            sample["palm_gyro"][0], sample["palm_gyro"][1], sample["palm_gyro"][2]
        )
        self.mgp.stdin.write(sample_line)
        self.sample_count += 1

        current_state = self.mgp.stdout.readline()
        s = list(map(lambda x: int(x), current_state.split()))
        if s[0] == 5:
            # Swing detected
            return self.sample_count - s[3]  # Length of followthrough
        return None


class E2EBaseRocket(E2EDetector):
    def get_classifier(self) -> LaunchpadClassifier:
        pass

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)

    def __init__(self,
                 name: str = "BaseRocket",
                 window_size: int = 40,
                 cooldown_period: int = 50,
                 palm_vibration_threshold: float = 2,
                 arm_gyro_x_threshold: float | None = 40,
                 palm_gyro_z_dif_threshold: float | None = None):
        self.name = name
        self.buffer = deque([])
        self.window_size = window_size
        self.window_period = (self.window_size // 4) * 3
        self.next_window = self.window_period
        self.sample_count = 0
        self.cooldown_period = cooldown_period
        self.cooldown_timer = 0
        self.active_followthroughs = []
        self.palm_vibration_threshold = palm_vibration_threshold
        self.arm_gyro_x_threshold = arm_gyro_x_threshold
        self.palm_gyro_z_dif_threshold = palm_gyro_z_dif_threshold

    def get_name(self) -> str:
        return self.name

    def iterate_impact(self) -> int | None:
        # Returns if this window contained an impact and if so, how many samples back was it
        self.next_window -= 1
        self.sample_count += 1
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            return None
        if self.next_window <= 0:
            self.next_window = self.window_period
            if len(self.buffer) < self.window_size:
                return None
            else:
                return detect_threshold(
                    list(self.buffer)[(len(self.buffer) - self.window_size):],
                    palm_vibration_threshold=self.palm_vibration_threshold,
                    arm_gyro_x_threshold=self.arm_gyro_x_threshold,
                    palm_gyro_z_dif_threshold=self.palm_gyro_z_dif_threshold
                )
        else:
            return None

    def add_sample(self, sample: WristSample) -> int | None:
        self.buffer.append(sample)
        if len(self.buffer) > 300:
            self.buffer.popleft()
        imp_result = self.iterate_impact()

        return_result = None
        ft_to_remove = []
        for idx, ft in enumerate(self.active_followthroughs):
            self.active_followthroughs[idx] -= 1
            if ft <= 0:
                ft_to_remove.append(idx)
                # Run ROCKET for final validation
                # Impact should be 100 samples before
                if self.get_classifier().is_swing(self.get_samples_for_rocket()):
                    if self.cooldown_timer <= 0:
                        return_result = self.sample_count - 100
                    self.cooldown_timer = self.cooldown_period

        ft_to_remove.sort(reverse=True)
        for r in ft_to_remove:
            self.active_followthroughs.pop(r)

        if imp_result is not None:
            self.active_followthroughs.append(100 - imp_result)

        return return_result


class E2ERocketAlpha(E2EBaseRocket):
    classifier: LaunchpadClassifier | None = None

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketAlpha.classifier

    def __init__(self):
        super().__init__(name="RocketAlpha")
        if E2ERocketAlpha.classifier is None:
            E2ERocketAlpha.classifier = RocketPuttingRidge("split_0.100_20220502_new_putts.pck")


class E2ERocketBeta(E2EBaseRocket):
    classifier: LaunchpadClassifier | None = None

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketBeta.classifier

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)[self.bufslice]

    def __init__(self):
        super().__init__(name="RocketBeta")
        self.bufslice = slice(150, 250)
        if E2ERocketBeta.classifier is None:
            E2ERocketBeta.classifier = RocketPuttingRidge(
                "split_0.100_20220502_new_putts.pck",
                crop=self.bufslice,
                dimensions_to_remove=["arm_acc_x", "arm_acc_y", "arm_acc_z",
                                      "palm_acc_x", "palm_acc_y", "palm_acc_z",
                                      "palm_gyro_y", "palm_gyro_z",
                                      "arm_gyro_y", "palm_gyro_z"],
                synthesize_dimensions=[PalmAccDifSynth(), ArmGyroNormSynth(), PalmGyroNormSynth()]
            )


class E2ERocketPuttingPrime(E2EBaseRocket):
    classifier_dict: dict[str, LaunchpadClassifier] = {}

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketPuttingPrime.classifier_dict[f"{self.name}+{self.split}"]

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)[self.crop]

    def __init__(self,
                 name: str,
                 split: str = _default_split,
                 crop: slice = slice(0, 300),
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = [],
                 window_size: int = 50):
        super().__init__(name=name,
                         arm_gyro_x_threshold=23,
                         palm_vibration_threshold=1.75,
                         window_size=window_size,
                         cooldown_period=50)
        self.crop = crop
        self.split = split
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions
        if f"{self.name}+{self.split}" not in E2ERocketPuttingPrime.classifier_dict:
            E2ERocketPuttingPrime.classifier_dict[f"{self.name}+{self.split}"] = RocketPuttingRidge(
                split,
                crop=self.crop,
                dimensions_to_remove=dimensions_to_remove,
                synthesize_dimensions=synthesize_dimensions
            )


class E2ERocketPuttingIsolation(E2EBaseRocket):
    classifier_dict: dict[str, LaunchpadClassifier] = {}

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketPuttingIsolation.classifier_dict[f"{self.name}+{self.split}"]

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)[self.crop]

    def __init__(self,
                 name: str,
                 split: str = _default_split,
                 crop: slice = slice(0, 300),
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = [],
                 window_size: int = 50):
        super().__init__(name=name,
                         arm_gyro_x_threshold=23,
                         palm_vibration_threshold=1.75,
                         window_size=window_size,
                         cooldown_period=50)
        self.crop = crop
        self.split = split
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions
        if f"{self.name}+{self.split}" not in E2ERocketPuttingIsolation.classifier_dict:
            E2ERocketPuttingIsolation.classifier_dict[f"{self.name}+{self.split}"] = RocketPuttingIsolation(
                split,
                crop=self.crop,
                dimensions_to_remove=dimensions_to_remove,
                synthesize_dimensions=synthesize_dimensions
            )


class E2ERocketFullSwingPrime(E2EBaseRocket):
    classifier_dict: dict[str, LaunchpadClassifier] = {}

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketFullSwingPrime.classifier_dict[f"{self.name}+{self.split}"]

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)[self.crop]

    def __init__(self,
                 name: str,
                 split: str = _default_split,
                 crop: slice = slice(0, 300),
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = [],
                 window_size: int = 80):
        super().__init__(name=name,
                         palm_vibration_threshold=6,
                         arm_gyro_x_threshold=None,
                         palm_gyro_z_dif_threshold=-100,
                         window_size=window_size,
                         cooldown_period=75)
        self.crop = crop
        self.split = split
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions
        if f"{self.name}+{self.split}" not in E2ERocketFullSwingPrime.classifier_dict:
            E2ERocketFullSwingPrime.classifier_dict[f"{self.name}+{self.split}"] = RocketFullSwingRidge(
                split,
                crop=self.crop,
                dimensions_to_remove=dimensions_to_remove,
                synthesize_dimensions=synthesize_dimensions
            )


class E2ERocketFullSwingIsolation(E2EBaseRocket):
    classifier_dict: dict[str, LaunchpadClassifier] = {}

    def get_classifier(self) -> LaunchpadClassifier:
        return E2ERocketFullSwingIsolation.classifier_dict[f"{self.name}+{self.split}"]

    def get_samples_for_rocket(self) -> list[WristSample]:
        return list(self.buffer)[self.crop]

    def __init__(self,
                 name: str,
                 split: str = _default_split,
                 crop: slice = slice(0, 300),
                 dimensions_to_remove: list[str] = [],
                 synthesize_dimensions: list[DimensionSynth] = [],
                 window_size: int = 50):
        super().__init__(name=name,
                         palm_vibration_threshold=6,
                         arm_gyro_x_threshold=None,
                         palm_gyro_z_dif_threshold=-100,
                         window_size=window_size,
                         cooldown_period=75)
        self.crop = crop
        self.split = split
        self.dimensions_to_remove = dimensions_to_remove
        self.synthesize_dimensions = synthesize_dimensions
        if f"{self.name}+{self.split}" not in E2ERocketFullSwingIsolation.classifier_dict:
            E2ERocketFullSwingIsolation.classifier_dict[f"{self.name}+{self.split}"] = RocketFullSwingIsolation(
                split,
                crop=self.crop,
                dimensions_to_remove=dimensions_to_remove,
                synthesize_dimensions=synthesize_dimensions
            )