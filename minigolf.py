from matplotlib import lines
from sensor_data_types import WristSample, DominantHand, WornHand
from typing import TypedDict
import enum
import os
import subprocess

class MinigolfDetector(enum.Enum):
    FULLSWING = 0
    PUTTING = 1

class MinigolfResult(TypedDict):
    address:    int
    top:        int
    impact:     int

class MinigolfConfig(TypedDict):
    dominantHand:   DominantHand
    wornHand:       WornHand
    detector:       MinigolfDetector

class MinigolfRun(TypedDict):
    config:     MinigolfConfig
    result:     MinigolfResult

def convert_to_minigolf(
    samples: list[WristSample], 
    dominantHand: DominantHand, 
    wornHand: WornHand, 
    detector: MinigolfDetector) -> list[str]:
    lines = [f"{dominantHand.value} {wornHand.value} {detector.value}\n"]
    
    for sample in samples:
        lines.append(
            "{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(
                sample["arm_acc"][0], sample["arm_acc"][1], sample["arm_acc"][2],
                sample["arm_gyro"][0], sample["arm_gyro"][1], sample["arm_gyro"][2],
                sample["palm_acc"][0], sample["palm_acc"][1], sample["palm_acc"][2],
                sample["palm_gyro"][0], sample["palm_gyro"][1], sample["palm_gyro"][2]
            )
        )
    return lines

def run(data: list[str]) -> MinigolfResult | None:
    mgp = subprocess.Popen("./minigolf",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1)

    # 1st line must be metadata and will have no reponse from minigolf
    # 1st line: 0 for left handed, 1 for right handed
    # 2nd line: 0 for trail wrist, 1 for lead wrist
    # 3rd line: 0 for full swing, 1 for putting
    # Then
    # 1st line Arm Acc x y z
    # 2nd line Arm Gyro x y z
    # 3rd line Palm Acc x y z
    # 4th line Palm Gyro x y z

    result = None

    for i, l in enumerate(data):
        if i == 0:
            mgp.stdin.write(l)
            continue

        mgp.stdin.write(l)

        current_state = mgp.stdout.readline()
        s = list(map(lambda x: int(x), current_state.split()))
        if s[0] == 5:
            # Swing detected
            result = MinigolfResult()
            result["address"] = i - s[1] - s[2] - s[3]
            result["top"] = i - s[2] - s[3]
            result["impact"] = i - s[3]
            break

    mgp.terminate()
    return result

def run_configs(
    samples: list[WristSample],
    configs: list[MinigolfConfig]) -> list[MinigolfRun]:
    results: list[MinigolfRun] = []
    for c in configs:
        this_run = MinigolfRun()
        this_run["config"] = c
        print(f" * Running with config {c}")
        mgd = convert_to_minigolf(samples, c["dominantHand"], c["wornHand"], c["detector"])
        this_run["result"] = run(mgd)
        
        results.append(this_run)
    return results

def run_full_configs(samples: list[WristSample]) -> list[MinigolfRun]:
    all_configs: list[MinigolfConfig] = [
        {
            "dominantHand": DominantHand.RIGHT,
            "wornHand": WornHand.OFFHAND,
            "detector": MinigolfDetector.FULLSWING
        },
        {
            "dominantHand": DominantHand.RIGHT,
            "wornHand": WornHand.OFFHAND,
            "detector": MinigolfDetector.PUTTING
        },
        {
            "dominantHand": DominantHand.RIGHT,
            "wornHand": WornHand.DOMINANT,
            "detector": MinigolfDetector.FULLSWING
        },
        {
            "dominantHand": DominantHand.RIGHT,
            "wornHand": WornHand.DOMINANT,
            "detector": MinigolfDetector.PUTTING
        },
        {
            "dominantHand": DominantHand.LEFT,
            "wornHand": WornHand.OFFHAND,
            "detector": MinigolfDetector.FULLSWING
        },
        {
            "dominantHand": DominantHand.LEFT,
            "wornHand": WornHand.OFFHAND,
            "detector": MinigolfDetector.PUTTING
        },
        {
            "dominantHand": DominantHand.LEFT,
            "wornHand": WornHand.DOMINANT,
            "detector": MinigolfDetector.FULLSWING
        },
        {
            "dominantHand": DominantHand.LEFT,
            "wornHand": WornHand.DOMINANT,
            "detector": MinigolfDetector.PUTTING
        },
    ]
    return run_configs(samples, all_configs)