import concurrent.futures
import subprocess
import torch


if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "metal"
else:
    device = "cpu"


def create_command(
    EPOCHS: int,
    SEED: int = None,
    device: str = device,
    LEARNING_RATE: float = None,
    BATCH_SIZE: int = None,
    CHECK_POINT: str = None,
    DROPOUT: bool = False,
    BATCHNORM: bool = False,
    DATA_AUGMENTATION: bool = False,
):
    command = "mlc model.train"
    command += " -t"
    command += f" --epoch {EPOCHS}"
    if SEED is not None:
        command += f" --seed {SEED}"
    if device is not None:
        command += f" -d {device}"
    if LEARNING_RATE is not None:
        command += f" -l {LEARNING_RATE}"
    if BATCH_SIZE is not None:
        command += f" -b {BATCH_SIZE}"

    # model commands
    command += " VGG"
    if DROPOUT:
        command += " --dropout"
    if BATCHNORM:
        command += " --batchnorm"

    # dataset commands
    command += " cats_and_dogs"
    if DATA_AUGMENTATION:
        command += " --data-augmentation"
    if CHECK_POINT is not None:
        command += f" --checkpoint {CHECK_POINT}"

    return command


def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)


# Common experiment grid
epochs = 10

# === EXPERIMENTS: Grouped into 4 batches ===

commands_1 = []  # Machine 1: dropout vs no-dropout
for dropout in [False, True]:
    commands_1.append(
        create_command(
            EPOCHS=epochs,
            DROPOUT=dropout,
        )
    )

commands_2 = []  # Machine 2: batchnorm vs no-batchnorm
for bn in [False, True]:
    commands_2.append(
        create_command(
            EPOCHS=epochs,
            BATCHNORM=bn,
        )
    )

commands_3 = []  # Machine 3: data augmentation vs none
for aug in [False, True]:
    commands_3.append(
        create_command(
            EPOCHS=epochs,
            DATA_AUGMENTATION=aug,
        )
    )


# === Execute each group separately ===
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run, commands_1)
