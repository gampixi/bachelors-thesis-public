import glob
import random
import string
import os

path_to = "./unprocessed_raw/"

bin_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.bin")]
cal_files = [x.split('/')[-1] for x in glob.glob(f"{path_to}*.cal")]

# print(bin_files)
# print(cal_files)

# Find matching pairs

pairs = []

# O(n^2) ewwwwww
for bix, b in enumerate(bin_files):
    common_part = b[0:-4]
    matching_cal_name = f"{common_part}_CD.cal"
    for idx, c in enumerate(cal_files):
        if c == matching_cal_name:
            print(f"Found pair: {c} {b}")
            pairs.append((b, c,))
            bin_files[bix] = "USED UP NAME"
            cal_files[idx] = "USED UP NAME"
            break

# Rename pairs

for p in pairs:
    print(f"Renaming {p}")
    random_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    new_bin_name = p[0].split("_")
    new_bin_name[0] = random_name
    new_cal_name = p[1].split("_")
    new_cal_name[0] = random_name
    nbn = "_".join(new_bin_name)
    ncn = "_".join(new_cal_name)
    os.rename(f"{path_to}{p[0]}", f"{path_to}{nbn}")
    os.rename(f"{path_to}{p[1]}", f"{path_to}{ncn}")

# Remove unrenamed files

print("Deleting unrenamed")

for b in bin_files:
    if b != "USED UP NAME":
        os.remove(f"{path_to}{b}")

for b in cal_files:
    if b != "USED UP NAME":
        os.remove(f"{path_to}{b}")