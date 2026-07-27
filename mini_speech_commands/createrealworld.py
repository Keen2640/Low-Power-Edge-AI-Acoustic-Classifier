import os, shutil, random

SOURCE = "mini_speech_commands"
DEST = "real_world_test"
CLASSES = ["yes", "no", "up"]
N_HELD_OUT = 40  # how many unseen clips per class to pull

for class_name in CLASSES:
    src_folder = os.path.join(SOURCE, class_name)
    dst_folder = os.path.join(DEST, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(src_folder) if f.endswith(".wav")])
    held_out = files[300:300 + N_HELD_OUT]  # skip the 300 used in training

    for f in held_out:
        shutil.copy(os.path.join(src_folder, f), os.path.join(dst_folder, f))

    print(f"{class_name}: copied {len(held_out)} held-out files")
